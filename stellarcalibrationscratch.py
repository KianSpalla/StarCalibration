
"""
Purpose of this script is to take a GONet image, find the stars in the image, match the stars to a star catalog, and then use the matched stars to calibrate the image.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import (
    label as nd_label,
    sum as ndimage_sum,
    center_of_mass as ndimage_com,
    shift as nd_shift,
)
from GONet_Wizard.GONet_utils import GONetFile
from PIL import Image
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.io import fits
from astroquery.gaia import Gaia

#Function that takes a mask and finds the stars in the mask using connected component labeling
#This is donne using the scipy.ndimage.label function which labels connected components in a binary image and returns the labels and the number of stars found
#We did this manually using the compact_search function. This was is easier, I havent tested effeciency but it should be faster than the compact_search function 
#which is a brute force method that checks every pixel in the image for a star. The connected component labeling method only checks the pixels that are above the threshold and groups them together as stars.
def find_stars(mask):
    labels, num_stars = nd_label(mask)
    return labels, num_stars

#Function that takes the label and number of stars and finds the centroids based on the brightness of the stars. This is done using the scipy.ndimage.center_of_mass function which calculates the center of mass of the labeled stars and returns the centroids as a list of (x, y) coordinates.
def find_centroids(labels, num_stars):
    centroids = np.array(ndimage_com(labels, labels, range(1, num_stars + 1)))
    return centroids

def catalog_query(meta, gmax, radius_deg):
    lat_deg = float(meta["GPS"]["latitude"])
    lon_deg = float(meta["GPS"]["longitude"])
    alt_m = float(meta["GPS"]["altitude"])
    ut_iso = meta["DateTime"].replace(":", "-", 2).replace(" ", "T")
    location = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg, height=alt_m * u.m)
    obsTime = Time(ut_iso, scale="utc")
    zenith_altaz = SkyCoord(
        alt=90 * u.deg,
        az=0  * u.deg,
        frame=AltAz(obstime=obsTime, location=location),
    )
    zenith_icrs = zenith_altaz.icrs
    Gaia.ROW_LIMIT = 200000           # safety cap: never download more than 200 k rows
    ra0  = zenith_icrs.ra.deg
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
    job = Gaia.launch_job(query)
    results = job.get_results()

    stars_icrs  = SkyCoord(ra  = np.array(results["ra"])  * u.deg,
                           dec = np.array(results["dec"]) * u.deg,
                           frame="icrs")
    stars_altaz = stars_icrs.transform_to(
        AltAz(obstime=obsTime, location=location)
    )

    alt  = stars_altaz.alt.deg
    az   = stars_altaz.az.deg
    gmag = np.array(results["phot_g_mean_mag"])

    above = alt > 0
    alt, az, gmag = alt[above], az[above], gmag[above]
    return alt, az, gmag

def unitvec_from_altaz(alt, az):
    alt_rad = np.deg2rad(alt)
    az_rad  = np.deg2rad(az)
    x = np.cos(alt_rad) * np.cos(az_rad)
    y = np.cos(alt_rad) * np.sin(az_rad)
    z = np.sin(alt_rad)
    return np.stack([x, y, z], axis=-1)

def rotate_z(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], float)

def rotate_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], float)

def rotate_x(angle_rad):    
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], float)

def orientation_matrix(alpha, beta, gamma):
    Rz_alpha = rotate_z(alpha)
    Ry_beta  = rotate_y(beta)
    Rz_gamma = rotate_z(gamma)
    return Rz_alpha @ Ry_beta @ Rz_gamma

def theta_from_r(r_pix, R_pix):
    return (r_pix / R_pix) * (np.pi / 2)

def r_from_theta(theta, R_pix):
    return (theta / (np.pi / 2)) * R_pix


def predict_pixels_from_catalog(alt, az, cx, cy, R_pix, alpha, beta, gamma):
    alt = np.deg2rad(alt)
    az  = np.deg2rad(az)

    v_sky = unitvec_from_altaz(alt, az)   # (M,3)

    R = orientation_matrix(alpha, beta, gamma)   # cam -> sky
    # We need sky -> cam, so invert (rotation matrix inverse = transpose)
    v_cam = v_sky @ R.T

    # Convert cam vector -> (theta, phi)
    # theta = angle from cam +z
    z = np.clip(v_cam[:, 2], -1, 1)
    theta = np.arccos(z)
    phi = np.arctan2(v_cam[:, 1], v_cam[:, 0])

    # Convert theta -> pixel radius
    r = r_from_theta(theta, R_pix)

    # Polar -> pixel
    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)
    return np.stack([x, y], axis=-1)

def match_score(img_xy, pred_xy, tol_pix):
    tree = cKDTree(pred_xy)
    dists, idx = tree.query(img_xy, distance_upper_bound=tol_pix)
    matched = dists < tol_pix
    score = np.sum(matched)
    return score, dists, idx

def solve_orientation(star_positions, alt, az, gmag, cx, cy, R_pix):
    img_tree = cKDTree(star_positions)
    alpha_grid = np.deg2rad(np.arange(0, 360, 1))
    beta_grid = np.deg2rad(np.arange(0, 90, 1))
    gamma_grid = np.deg2rad(np.arange(0, 360, 1))

    best = {"score": -1}

    for beta in beta_grid:
        # If beta is 0, gamma doesn’t matter — you can skip gamma loop for speed
        gamma_list = [0.0] if abs(beta) < 1e-12 else gamma_grid

        for gamma in gamma_list:
            for alpha in alpha_grid:
                pred_xy = predict_pixels_from_catalog(
                    alt, az,
                    cx, cy, R_pix,
                    alpha, beta, gamma
                )
                s, dists, idx = match_score(star_positions, pred_xy, tol_pix=25.0)
                if s > best["score"]:
                    best = {
                        "score": s,
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "dists": dists,
                        "idx": idx,
                        "pred_xy": pred_xy
                    }
    return best

def fit_wcs_and_center_zenith(
    sub,
    img_xy,
    cat_alt_deg,
    cat_az_deg,
    best,
    meta,
    tol_pix=25.0,
    min_wcs_matches=3,):
    matched_catalog_mask = best["dists"] <= tol_pix
    matched_catalog_indices = np.where(matched_catalog_mask)[0]
    matched_image_indices = best["idx"][matched_catalog_mask]
    matched_distances_pixels = best["dists"][matched_catalog_mask]



def calibrateImage(image_path):
    #Load the image and extract the green channel
    go = GONetFile.from_file(image_path)
    img = go.green

    #Change variable to find more stars
    N = 5

    #Create a mask to find stars based on a threshold using the mean and standard deviation of the image
    mean = np.mean(img)
    std = np.std(img)
    threshold = mean + N * std
    mask = img > threshold
    labels, num_stars = find_stars(mask)
    star_positions = find_centroids(labels, num_stars)

    cx, cy             = 1030, 760
    R_pix              = 740
    catalog_radius_deg = 60.0

    metadata = go.meta
    gmax = 2.5
    alt, az, gmag = catalog_query(metadata, gmax, catalog_radius_deg)

    best = solve_orientation(star_positions, alt, az, gmag, cx, cy, R_pix)

    #plot the image and star positions
    plt.imshow(img)
    plt.scatter(star_positions[:, 1], star_positions[:, 0], s=10, edgecolor='red', facecolor='none')
    plt.title('Detected Stars')
    plt.show()


def main():
    image_path = r'C:\Users\spall\Documents\GitHub\StarCalibration\Testing Images\256_251029_204008_1761770474.jpg'
    calibrateImage(image_path)

if __name__ == "__main__":  
    main()