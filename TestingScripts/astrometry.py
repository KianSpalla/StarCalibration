#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

from astroquery.gaia import Gaia

from GONet_Wizard.GONet_utils import GONetFile

def compute_cat_unit_coords(image_path: Path, radius_deg: float = 60.0, gmax: float = 8.0, top_m: int = 1000) -> np.ndarray:
    go = GONetFile.from_file(Path(image_path))
    go.remove_overscan()
    img_gray = go.green  # not used directly but validates image load
    meta = go.meta

    # Extract metadata
    lat_deg = float(meta['GPS']['latitude'])
    lon_deg = float(meta['GPS']['longitude'])
    alt_m = float(meta['GPS']['altitude'])
    ut_iso = meta['DateTime'].replace(':', '-', 2).replace(' ', 'T')

    # Location/time
    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    t = Time(ut_iso, scale="utc")

    # Zenith in ICRS
    zenith_altaz = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=AltAz(obstime=t, location=location))
    zenith_icrs = zenith_altaz.icrs

    # Gaia query
    Gaia.ROW_LIMIT = 200000
    ra0 = zenith_icrs.ra.deg
    dec0 = zenith_icrs.dec.deg
    query = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra0}, {dec0}, {radius_deg})
    )
    AND phot_g_mean_mag < {gmax}
    """
    job = Gaia.launch_job_async(query)
    tbl = job.get_results()

    # Build SkyCoord and transform to Alt/Az
    stars_icrs = SkyCoord(ra=np.array(tbl["ra"]) * u.deg, dec=np.array(tbl["dec"]) * u.deg, frame="icrs")
    altaz_frame = AltAz(obstime=t, location=location)
    stars_altaz = stars_icrs.transform_to(altaz_frame)

    alt = stars_altaz.alt.deg
    az = stars_altaz.az.deg
    gmag = np.array(tbl["phot_g_mean_mag"])

    # Above horizon
    above = alt > 0
    alt = alt[above]
    az = az[above]
    gmag = gmag[above]

    # Keep top_m brightest (smallest mag) if desired
    if top_m is not None and len(gmag) > top_m:
        idx = np.argsort(gmag)[:top_m]
        alt = alt[idx]
        az = az[idx]
        gmag = gmag[idx]

    # Project to unit disk
    theta_deg = 90.0 - alt
    r = theta_deg / 90.0
    az_rad = np.deg2rad(az)
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)

    cat_xy_unit = np.column_stack((x, y)).astype(float)
    return cat_xy_unit

def main():
    parser = argparse.ArgumentParser(description="Compute catalog unit-disk coords (cat_xy_unit) from image metadata.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--radius", type=float, default=60.0, help="Gaia cone radius in degrees")
    parser.add_argument("--gmax", type=float, default=8.0, help="Magnitude cut (phot_g_mean_mag < gmax)")
    parser.add_argument("--top_m", type=int, default=1000, help="Limit to top M brightest above horizon")
    parser.add_argument("--save", action="store_true", help="Save cat_xy_unit .npy next to image")
    args = parser.parse_args()

    cat_xy_unit = compute_cat_unit_coords(Path(args.image), radius_deg=args.radius, gmax=args.gmax, top_m=args.top_m)
    print(f"Computed {len(cat_xy_unit)} catalog unit coords.")

    if args.save:
        out_path = Path(args.image).with_suffix("")
        out_file = out_path.parent / f"{out_path.name}_cat_xy_unit.npy"
        np.save(out_file, cat_xy_unit)
        print(f"Saved cat_xy_unit to: {out_file}")
    else:
        print("cat_xy_unit (preview):")
        print(cat_xy_unit[:5])

if __name__ == "__main__":
    main()