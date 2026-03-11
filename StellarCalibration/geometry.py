import numpy as np

def unitvec_from_altaz(alt_rad, az_rad):
    ca = np.cos(alt_rad)
    x = ca * np.sin(az_rad)
    y = ca * np.cos(az_rad)
    z = np.sin(alt_rad)
    return np.stack([x, y, z], axis=-1)


def rot_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def orientation_matrix(alpha, beta, gamma):
    R_alpha = rot_z(alpha)
    R_tilt = rot_z(gamma) @ rot_x(beta) @ rot_z(-gamma)
    return R_tilt @ R_alpha


def r_from_theta(theta_rad, radiusPix):
    return (theta_rad / (np.pi / 2)) * radiusPix


def filter_image_sources_by_radius(imgXY, cx, cy, radiusPix, radiusDeg):
    if len(imgXY) == 0:
        return imgXY

    maxRadiusPix = (float(radiusDeg) / 90.0) * float(radiusPix)
    dx = imgXY[:, 0] - float(cx)
    dy = imgXY[:, 1] - float(cy)
    rr = np.sqrt(dx * dx + dy * dy)
    keep = rr <= maxRadiusPix
    return imgXY[keep]


def predict_pixels_from_catalog(altDeg, azDeg, cx, cy, radiusPix, alpha, beta, gamma):
    alt = np.deg2rad(altDeg)
    az = np.deg2rad(azDeg)

    v_sky = unitvec_from_altaz(alt, az)
    R = orientation_matrix(alpha, beta, gamma)
    v_cam = v_sky @ R.T

    z = np.clip(v_cam[:, 2], -1, 1)
    theta = np.arccos(z)
    phi = np.arctan2(v_cam[:, 1], v_cam[:, 0])

    r = r_from_theta(theta, radiusPix)
    x = cx + r * np.cos(phi)
    y = cy + r * np.sin(phi)
    return np.stack([x, y], axis=-1)