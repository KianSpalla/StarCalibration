
# === Imports ===
# TODO: import numpy, matplotlib.pyplot, and an image loading function (e.g., imageio.v3 or plt.imread)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from GONet_Wizard.GONet_utils import GONetFile
#plt.ion()

# ---------------------------------------------------------------------
# EXERCISE 0.1 — Load an image
# ---------------------------------------------------------------------
# 1. Choose a GONet image file (set a Path pointing to it).
# 2. Load the image
# 3. Save one of the channes to a variable, e.g. `img`
# 4. Print:
#    - type(img)
#    - img.shape
#    - img.dtype
#
# Hints:
# - Use GONetFile.from_file(path).
#
image_path = Path(r"Testing Images/256_251029_204008_1761770474.jpg")
go = GONetFile.from_file(image_path)
go.remove_overscan()
img = go.green
#print(type(img))
#print(img.shape)
#print(img.dtype)



# ---------------------------------------------------------------------
# EXERCISE 0.2 — Convert to grayscale (if needed)
# ---------------------------------------------------------------------
# Visualize the image
#
# Hints:
# - Display using plt.imshow
#plt.imshow(img,cmap='gray')
#plt.show()
# print(img.shape[0]/2)
# print(img.shape[1]/2)
# ---------------------------------------------------------------------
# EXERCISE 1.1 — Summary statistics
# ---------------------------------------------------------------------
# Compute and print the following for img:
# - minimum value
# - maximum value
# - mean value
# - standard deviation
#   img.min(), img.max(), img.mean(), img.std()

img_ravel = np.ravel(img)

min_val = np.min(img_ravel)
max_val = np.max(img_ravel)
mean_val = np.mean(img_ravel)
std_val = np.std(img_ravel)
#print("min:", min_val)
#print("max:", max_val)
#print("mean:", mean_val)
#print("std:", std_val)


# ---------------------------------------------------------------------
# EXERCISE 1.2 — Histogram of pixel values
# ---------------------------------------------------------------------
# Plot a histogram of all pixel values in img.
#
# Steps:
# - Flatten the image with img.ravel()
# - Use plt.hist(..., bins=100)
# - Label the x-axis as "Pixel value" and y-axis as "Count"

# plt.figure()
# plt.hist(img_ravel, bins=100)
# plt.xlabel("Pixel value")
# plt.ylabel("Count")
# plt.title("Histogram of pixel values")
#plt.show()
# Commented out plotting calls
plt.figure()
plt.hist(img_ravel, bins=1000)
plt.xlabel("Pixel value")
plt.ylabel("Count")
plt.title("Histogram of pixel values")
plt.show()


# ---------------------------------------------------------------------
# EXERCISE 2.1 — Extract a sub-image (cutout)
# ---------------------------------------------------------------------
# Choose a rectangular region of the image that contains stars.
# For example, pick rows [y0:y1] and columns [x0:x1].
#
# Steps:
# - Define y0, y1, x0, x1.
# - Extract sub = img[y0:y1, x0:x1]
# - Print sub.shape.
# - Display sub with imshow + colorbar.
#
y0, y1 = 500, 700    # example; adjust to where stars are
x0, x1 = 500, 700
sub = img[y0:y1, x0:x1]
#print("sub shape:", sub.shape)

# plt.figure()
# plt.imshow(sub, origin="lower")
# plt.colorbar()
# plt.title("Sub-image (cutout)")
#plt.show()
# Commented out plotting calls
# plt.figure()
# plt.imshow(sub, origin="lower")
# plt.colorbar()
# plt.title("Sub-image (cutout)")


# ---------------------------------------------------------------------
# EXERCISE 2.2 — Adjust contrast on the sub-image
# ---------------------------------------------------------------------
# We want to change vmin and vmax to better see stars.
#
# Steps:
# - Compute mean and std of sub: sub_mean, sub_std.
# - Display sub with:
#       vmin = sub_mean - 1 * sub_std
#       vmax = sub_mean + 5 * sub_std
# - Experiment: try changing the multipliers (e.g. 0.5, 2, 10) and see how
#   the visibility of stars changes.
#
sub_ravel=np.ravel(sub)
#print(sub)
sub_mean = np.mean(sub_ravel)
sub_std = np.std(sub_ravel)
#
# plt.figure()
# plt.imshow(sub, origin="lower",
#            vmin=sub_mean - 1 * sub_std,
#            vmax=sub_mean + 5 * sub_std)
# plt.colorbar()
# plt.title("Sub-image with adjusted contrast")
#plt.show()
# Commented out plotting calls
# plt.figure()
# plt.imshow(sub, origin="lower",
#            vmin=sub_mean - 1 * sub_std,
#            vmax=sub_mean + 5 * sub_std)
# plt.colorbar()
# plt.title("Sub-image with adjusted contrast")


# ---------------------------------------------------------------------
# EXERCISE 3.1 — Create a threshold mask for star pixels
# ---------------------------------------------------------------------
# We want to mark "bright" pixels as candidate star pixels using a simple rule:
#
#   threshold = sub_mean + N * sub_std
#
# where N is something like 5 (you can try 3, 5, 7, etc.).
#
# Steps:
# - Choose N (e.g. N = 5).
# - Compute threshold.
# - Create a boolean mask:
#       mask = sub > threshold
# - Display mask using imshow(mask, origin="lower", cmap="gray").
#
N = 5
threshold = sub_mean+N*sub_std
mask = sub>threshold
# plt.figure()
# plt.imshow(mask, origin="lower", cmap="gray")
# plt.title(f"Mask of star pixels (N = {N})")
#plt.show()
# Commented out plotting calls
# plt.figure()
# plt.imshow(mask, origin="lower", cmap="gray")
# plt.title(f"Mask of star pixels (N = {N})")


# ---------------------------------------------------------------------
# EXERCISE 3.2 — Count star pixels and fraction
# ---------------------------------------------------------------------
# Now we want to know how many pixels are marked as "star pixels".
#
# Steps:
# - Count them using np.sum(mask).
# - Compute the fraction of pixels above the threshold using mask.mean().
# - Print both.
#
num_star_pixels = np.sum(mask)
fraction_star_pixels = num_star_pixels/mask.mean()
#print("Number of star pixels:", num_star_pixels)
#print("Fraction of pixels above threshold:", fraction_star_pixels)

# Try changing N (for example 3, 5, 7) and see how these numbers change.


# ---------------------------------------------------------------------
# EXERCISE 4.1 — Get coordinates of star pixels
# ---------------------------------------------------------------------
# We want the (x, y) coordinates of all pixels where mask is True.
#
# Steps:
# - Use np.where(mask) → returns (ys, xs).
# - Print how many coordinates you got (should match num_star_pixels).
#
ys, xs = np.where(mask)
#print("Number of coordinates:", len(xs))


# ---------------------------------------------------------------------
# EXERCISE 4.2 — Overlay star pixels on the sub-image
# ---------------------------------------------------------------------
# Finally, let's visualize which pixels we detected as "star pixels" on top
# of the original sub-image.
#
# Steps:
# - Make a new figure.
# - Show sub with imshow (using vmin/vmax like before).
# - Use plt.scatter(xs, ys, ...) to overlay small circles at the star pixels.
#   (remember: xs = columns, ys = rows)
#
# Example:
#   plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
#
# plt.figure()
# plt.imshow(sub, origin="lower",
#            vmin=sub_mean - 1 * sub_std,
#            vmax=sub_mean + 5 * sub_std)
# plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
# plt.title("Detected star pixels")
# plt.colorbar()
#plt.show()
# Commented out plotting calls
# plt.figure()
# plt.imshow(sub, origin="lower",
#            vmin=sub_mean - 1 * sub_std,
#            vmax=sub_mean + 5 * sub_std)
# plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
# plt.title("Detected star pixels")
# plt.colorbar()


# ---------------------------------------------------------------------
# OPTIONAL EXERCISE 5 — Experiment with different thresholds
# ---------------------------------------------------------------------
# Try running the mask + overlay steps multiple times with different N values
# (e.g. 3, 4, 5, 6, 7) and observe:
# - How does the number of star pixels change?
# - Do you start picking up too much noise when N is small?
# - Do you start missing fainter stars when N is large?
#
# You can wrap the detection and plotting into a small function, like:
#
def show_star_pixels(sub, N):
    """
    For a given sub-image and threshold factor N, detect star pixels and plot.
    """
    sub_ravel=np.ravel(sub)
    sub_mean = np.mean(sub_ravel)
    sub_std=np.std(sub_ravel)
    # plt.figure()
    # plt.imshow(sub, origin="lower",
    #            vmin=sub_mean - 1 * sub_std,
    #            vmax=sub_mean + 5 * sub_std)
    # plt.scatter(xs, ys, s=10, edgecolors='red', facecolors='none')
    # plt.title(f'Threshold Factor: {N}')
    # plt.colorbar()
    #plt.show()

    
    
# #     # TODO: compute mean/std, threshold, mask, then plot sub + star pixels
# #     pass

# # # Then call:
# # show_star_pixels(sub, 3)
# # show_star_pixels(sub, 5)
# # show_star_pixels(sub, 7)
# # #
# # # End of exercise script.
mask_two=[
   [False, False, True,  False, False],
   [False, False, True,  False, False],
   [True,  True,  True,  False, False],
   [False, False, False, False, True ],
   [False, False, False, False, True ],
 ]
neighbor=[(-1,0),(1,0),(0,-1),(0,1)]
def compact_search(mask):
    height,width=mask.shape
    labels=np.zeros_like(mask,dtype=int)
    current_label=0
    for h in range(height):
        for w in range(width):
            if mask[h,w]==0:
                continue
            if labels[h,w]!=0:
                continue
            if mask[h,w]==1:
                current_label+=1
                compact_object=[(h,w)]
                labels[h,w]=current_label
                for n in neighbor:
                    while len(compact_object)>0:
                        current_element=compact_object.pop()
                        for n in neighbor:
                            nh=current_element[0]+n[0]
                            nw=current_element[1]+n[1]
                            if nh<0 or nh>=height or nw<0 or nw>=width:
                                continue
                            if mask[nh,nw]==1 and labels[nh,nw]==0:
                                labels[nh,nw]=current_label
                                compact_object.append([nh,nw])
    return labels,current_label
labels,num_clusters=compact_search(np.array(mask))
# plt.figure()
# plt.imshow(sub)
#plt.show()

for cluster in range(2,num_clusters+1):
    yc,xc=np.where(labels==cluster)
    star=sub[yc,xc]
    #print(star)
    peak=np.argmax(star)#returns the location of the max
    flux=np.sum(star)
    wx=np.sum(xc*star) /flux #follows centroid formula
    wy=np.sum(yc*star) /flux
    #print(peak)
    y_peak,x_peak=yc[peak],xc[peak]
    # plt.plot(wx,wy,'r.')
    # plt.title('Stars')
    
    #plt.imshow(base,alpha=0.5)
# input()
                
    