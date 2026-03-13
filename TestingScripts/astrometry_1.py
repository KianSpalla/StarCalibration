#!/usr/bin/env python3
"""
Exercise: From GPS + UT time to a sky catalog plot (side-by-side with the image)

What you will do (high level)
-----------------------------
1) Use GPS + UT time → compute zenith RA/Dec (roughly image center).
2) Query a star catalog (Gaia) around that zenith position.
3) Understand coordinate systems:
   - Catalog: RA/Dec (fixed on the celestial sphere)
   - Local sky: Alt/Az (depends on GPS + time)
   - Image: pixel (x, y) (cartesian coordinates on a flat sensor)
4) Convert catalog RA/Dec → Alt/Az at your observation.
5) Project Alt/Az into a simple zenith-centered fisheye plot (approximate).
6) Plot the catalog sky and your image side by side.

IMPORTANT NOTES
---------------
- This script requires internet access to query Gaia via astroquery.
- We are NOT doing "matching" yet (no WCS fit, no overlay of centroids).
- We will use an approximate fisheye projection for visualization only.

You will fill in TODO sections step by step.
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
# TODO: import what we need:
# - Path from pathlib
# - numpy
# - matplotlib.pyplot
# - astropy: Time, EarthLocation, AltAz, SkyCoord, units as u
# - astroquery Gaia
# - image loader
#
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

from astroquery.gaia import Gaia

from GONet_Wizard.GONet_utils import GONetFile



# ---------------------------------------------------------------------
# Step 0 — Load a GONet image and explore metadata
# ---------------------------------------------------------------------
# TODO:
# 1. Load a GONet image with GONetFile (use your project import path).
# 2. Access metadata with:
#       meta = go.meta
# 3. Explore:
#       type(meta)
#       meta.keys()
# 4. Find and extract:
#       latitude (deg)
#       longitude (deg)
#       altitude (meters)
#       UT time (UTC timestamp string)
#
#
# --- You must replace these with real values from meta ---
# lat_deg = ...
# lon_deg = ...
# alt_m   = ...
# ut_iso  = ...  # example format: "2025-12-13T03:12:45"
#
# Tip: If UT time in metadata is not ISO format, write a small conversion step.
#Value of meta ={'ResolutionUnit': 2, 'ExifOffset': 364, 'Make': 'RaspberryPi', 'Model': 'RP_imx477', 'DateTime': '2025:10:29 20:41:14', 'hostname': 'GONet256', 'version': '21.05', 'shutter_speed': -4.90689, 'DateTimeOriginal': '2025:10:29 20:41:14', 'DateTimeDigitized': '2025:10:29 20:41:14', 'bayer_width': 4056, 'bayer_height': 3040, 'ExifInteroperabilityOffset': 974, 'exposure_time': 29.999993, 'ISOSpeedRatings': 6, 'WhiteBalance': 0, 'analog_gain': 2048.0, 'image_width': 2028, 'image_height': 1520, 'GPS': {'latitude': 54.025555555555556, 'longitude': -9.822777777777777, 'altitude': 41.73}, 'JPEG': {'YCbCrPositioning': 'Centered', 'WB': [3.35, 1.59], 'ComponentsConfiguration': 'YCbCr', 'ColorSpace': 'sRGB'}}
image_path = Path(r"Testing Images/256_251029_204008_1761770474.jpg")
go = GONetFile.from_file(image_path)
go.remove_overscan()
img = go.green
meta = go.meta
# print(meta)
lat_deg = meta['GPS']['latitude']
lon_deg = meta['GPS']['longitude']
alt_m = meta['GPS']['altitude']
# Convert datetime from EXIF format (YYYY:MM:DD HH:MM:SS) to ISO format (YYYY-MM-DDTHH:MM:SS)
ut_iso = meta['DateTime'].replace(':', '-', 2).replace(' ', 'T')


# ---------------------------------------------------------------------
# Step 0.5 — Load the image pixels for plotting
# ---------------------------------------------------------------------
# TODO:
# 1. Read the image into an array called img.
# 2. Convert to grayscale if needed (if it has 3 channels).
#
img_gray = img  # GONetFile green channel is already grayscale



# ---------------------------------------------------------------------
# Step 1 — Compute zenith RA/Dec from GPS + UT time
# ---------------------------------------------------------------------
# Concepts:
# - Zenith means "straight up" at your location and time.
# - Zenith depends on WHERE you are (GPS) and WHEN (UT time).
# - We'll compute zenith in local Alt/Az as Alt=90°, then convert to RA/Dec.
#
# TODO:
# 1. Build an EarthLocation from lat/lon/alt.
# 2. Build a Time object from ut_iso.
# 3. Define a SkyCoord at Alt=90°, Az=0° in the AltAz frame.
#    (Az doesn't matter at the zenith.)
# 4. Convert that coordinate to ICRS (RA/Dec).
# 5. Print RA/Dec nicely.
#
location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
t = Time(ut_iso, scale="utc")

zenith_altaz = SkyCoord(
    alt=90 * u.deg,
    az=0 * u.deg,
    frame=AltAz(obstime=t, location=location)
)
zenith_icrs = zenith_altaz.icrs

# print("\n=== Zenith ===")
# print("Time (UTC):", t.isot)
# print("Zenith RA :", zenith_icrs.ra.to_string(unit=u.hour, sep=":"))
# print("Zenith Dec:", zenith_icrs.dec.to_string(unit=u.deg, sep=":"))
# print("============\n")


# ---------------------------------------------------------------------
# Step 2 — Query a catalog (Gaia) around the zenith
# ---------------------------------------------------------------------
# Concepts:
# - Gaia stores star positions in RA/Dec.
# - Gaia returns MANY stars. We must use:
#     (a) a cone radius around zenith
#     (b) a magnitude cut to reduce the number
#
# Magnitudes (explain to yourself / in comments):
# - Magnitude is a brightness scale.
# - Smaller number = brighter star.
# - Difference of 5 mag = factor of 100 in brightness.
#
# TODO:
# 1. Choose a search radius (deg). For a 2π camera it's up to 90°, but start smaller!
#    Suggested: 60° first (fewer stars), then 90° later.
# 2. Choose a magnitude cut (Gmax). Suggested: 8, 9, 10, 11.
# 3. Use an ADQL query to fetch:
#       ra, dec, phot_g_mean_mag
# 4. Print how many stars you got.
#
radius_deg = 60.0   # start smaller than 90
Gmax = 2.5      # start bright

# # Protect yourself with a row limit
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
AND phot_g_mean_mag < {Gmax}
"""

print("Submitting Gaia query...")
job = Gaia.launch_job_async(query)
tbl = job.get_results()

# print("\n=== Gaia query result ===")
# print("Radius (deg):", radius_deg)
# print("Gmax        :", Gmax)
# print("Returned rows:", len(tbl))
# print("=========================\n")

# Investigate magnitude range
gmag = np.array(tbl["phot_g_mean_mag"])
# print("G magnitude range:", np.nanmin(gmag), "to", np.nanmax(gmag))


# ---------------------------------------------------------------------
# Step 3 — RA/Dec vs Alt/Az vs pixels (concept checkpoint)
# ---------------------------------------------------------------------
# Write short answers as comments:
#
# Q1: Why are RA/Dec considered "fixed" for stars (ignoring proper motion)?
# Q2: Why do Alt/Az depend on GPS location and time?
# Q3: Why are our star detections (x, y) "cartesian" coordinates?
#
# A1:
# A2:
# A3:


# ---------------------------------------------------------------------
# Step 4 — Convert catalog RA/Dec → Alt/Az for our observation
# ---------------------------------------------------------------------
# Concepts:
# - Catalog gives RA/Dec (sky coordinates).
# - Our camera sees the local sky: stars have Alt/Az at our location/time.
# - We convert using an AltAz frame in Astropy.
#
# TODO:
# 1. Build a SkyCoord array from tbl["ra"], tbl["dec"] (degrees).
# 2. Transform to AltAz at time t and location.
# 3. Keep only stars above the horizon (alt > 0°).
# 4. Print how many are above horizon.

stars_icrs = SkyCoord(ra=np.array(tbl["ra"]) * u.deg,
                      dec=np.array(tbl["dec"]) * u.deg,
                      frame="icrs")

altaz_frame = AltAz(obstime=t, location=location)
stars_altaz = stars_icrs.transform_to(altaz_frame)

alt = stars_altaz.alt.deg
az  = stars_altaz.az.deg

above = alt > 0
alt = alt[above]
az  = az[above]
gmag = gmag[above]

# print("Stars above horizon:", len(alt))


# ---------------------------------------------------------------------
# Step 5 — Project Alt/Az into a zenith-centered fisheye plot
# ---------------------------------------------------------------------
# Concepts:
# - Our camera is approximately zenith-centered: zenith ~ image center.
# - Alt=90° (zenith) should map to plot center.
# - Alt=0° (horizon) should map to plot edge.
# - A fisheye lens has distortion, but for now we use a simple model.
#
# Simple projection for visualization (equidistant fisheye):
# - theta = angular distance from zenith = 90° - Alt
# - r = theta / (90°)  so r=0 at zenith, r=1 at horizon
# - Convert azimuth to x,y:
#     x = r * sin(az)
#     y = r * cos(az)
#
# TODO:
# 1. Compute theta_deg = 90 - alt.
# 2. Convert theta_deg to r in [0,1].
# 3. Convert az to radians.
# 4. Compute x, y.
# 5. Optionally scale marker sizes by magnitude (brighter = bigger).
#
theta_deg = 90.0 - alt
r = theta_deg / 90.0

az_rad = np.deg2rad(az)
x = r * np.sin(az_rad)
y = r * np.cos(az_rad)

# Optional size scaling (very rough):
# smaller mag => brighter => larger marker
sizes = np.clip(80 * 10 ** (-(gmag - np.nanmin(gmag)) / 2.5), 2, 120)


# ---------------------------------------------------------------------
# Step 5.5 — Lens center and distortions (concept checkpoint)
# ---------------------------------------------------------------------
# Write short answers as comments:
#
# Q4: Why do we need the image center to compare catalog stars to image stars?
# Q5: Why do lens distortions matter for mapping the sky to pixels?
#
# A4:
# A5:


# ---------------------------------------------------------------------
# Step 6 — Plot side-by-side: image vs catalog sky
# ---------------------------------------------------------------------
# Concepts:
# - At this stage we expect differences (rotation, flip, distortion).
# - Side-by-side is for intuition: does the star density/pattern look similar?
#
# TODO:
# 1. Make a figure with 1 row, 2 columns.
# 2. Left: show img_gray.
# 3. Right: scatter plot x,y with a circle boundary (r=1 is horizon).
# 4. Add titles that show radius_deg and Gmax.

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # Left: image
# ax1.imshow(img_gray, origin="lower")
# ax1.set_title("GONet image")
# ax1.set_axis_off()

# # Right: catalog fisheye
# ax2.scatter(x, y, s=sizes, alpha=0.7)
# horizon = plt.Circle((0, 0), 1.0, fill=False)
# ax2.add_patch(horizon)
# ax2.set_aspect("equal", "box")
# ax2.set_xlim(-1.05, 1.05)
# ax2.set_ylim(-1.05, 1.05)
# ax2.set_title(f"Gaia stars above horizon\n(radius={radius_deg}°, G<{Gmax})")
# ax2.set_xlabel("Projected x")
# ax2.set_ylabel("Projected y")

# plt.tight_layout()
# plt.show()


# ---------------------------------------------------------------------
# Step 7 — Exploration: how many stars vs magnitude cut?
# ---------------------------------------------------------------------
# TODO:
# Try a small experiment by rerunning your query with:
#   Gmax = 8, 9, 10, 11, 12
# and record:
# - total returned rows
# - number above horizon
#
# Write your results here as comments:
#
# Gmax=8  : returned=____ , above_horizon=____
# Gmax=9  : returned=____ , above_horizon=____
# Gmax=10 : returned=____ , above_horizon=____
# Gmax=11 : returned=____ , above_horizon=____
# Gmax=12 : returned=____ , above_horizon=____
#
# Observation:
# - At what Gmax does the star count become too large to be convenient?
# - What might that imply about the limiting magnitude of our camera?


# ---------------------------------------------------------------------
# End notes
# ---------------------------------------------------------------------
# Next lessons:
# - Overlay the *detected image centroids* on top of the projected catalog
# - Solve for rotation/flip and a better projection model
# - Fit a proper WCS / astrometric solution
