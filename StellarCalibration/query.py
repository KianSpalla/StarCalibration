import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astroquery.gaia import Gaia

def query_catalog_altaz_from_meta(meta, radius_deg=60.0, gmax=2.5, top_m=None):
    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m = float(meta["GPS"]["altitude"])
    ut_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")

    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    obstime = Time(ut_iso, scale="utc")

    zenith_altaz = SkyCoord(
        alt=90 * u.deg,
        az=0 * u.deg,
        frame=AltAz(obstime=obstime, location=location),
    )
    zenith_icrs = zenith_altaz.icrs

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

    stars_icrs = SkyCoord(ra=np.array(tbl["ra"]) * u.deg, dec=np.array(tbl["dec"]) * u.deg, frame="icrs")
    stars_altaz = stars_icrs.transform_to(AltAz(obstime=obstime, location=location))

    alt = stars_altaz.alt.deg
    az = stars_altaz.az.deg
    gmag = np.array(tbl["phot_g_mean_mag"])

    above = alt > 0
    alt, az, gmag = alt[above], az[above], gmag[above]

    if top_m is not None and len(gmag) > top_m:
        idx = np.argsort(gmag)[:top_m]
        alt, az, gmag = alt[idx], az[idx], gmag[idx]

    return alt, az, gmag