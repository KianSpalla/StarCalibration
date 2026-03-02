from GONet_Wizard.GONet_utils import GONetFile
import matplotlib.pyplot as plt
import numpy as np

#assign image that we want from GONetFile
go = GONetFile.from_file(r"C:\Users\spall\Desktop\GONet\Testing Images\256_251029_204008_1761770474.tiff")

##create an empty figure 
fig = plt.figure()

##add a subplot "ax" at the first position in a 1x1 grid 
#ax1 = fig.add_subplot(221)
#ax2 = fig.add_subplot(222)
#ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(111)

#row = go.green[0:len(go.green[1]), 10:20]
row = go.green[0, 10:20]
print("Mean of entire image before removal: ", go.green.mean())
print("Calculated Mean of strip: ", row.mean())

GreenImage = go.green

GreenImage = GreenImage - row.mean()
go.remove_overscan()
print("Mean using remove_overscan:", go.green.mean())
print("Mean after manual removal:", GreenImage.mean())
print("Difference between manual removal versus function remove_overscan():", np.abs(go.green.mean() - GreenImage.mean()))

#GreenImage = GreenImage.ravel()

median = np.median(GreenImage[600:700, 850:950])
stddev = np.std(GreenImage[600:700, 850:950])

plt.hist(GreenImage[600:700, 850:950].ravel(), bins=1000)

ax4.axvline(median, color='r', linestyle='dashed', linewidth=1)
ax4.axvline(median + stddev * 5, linestyle='dashed', linewidth=1, color='g')

#mask1 = GreenImage[600:700, 850:950] > (median + stddev * 2)
#mask2 = GreenImage[600:700, 850:950] > (median + stddev * 3)
#mask3 = GreenImage[600:700, 850:950] > (median + stddev * 1)
mask4 = GreenImage[600:700, 850:950] > (median + stddev * 5)

#ax1.imshow(mask1)
#ax1.set_title("Mask > median + 2*stddev")
#ax2.imshow(mask2)
#ax2.set_title("Mask > median + 3*stddev")
#ax3.imshow(mask3)
#ax3.set_title("Mask > median + 1*stddev")
ax4.imshow(mask4)
ax4.set_title("Mask > median + 5*stddev")

pixel_radius = 10
coords = np.argwhere(mask4)  # all pixels where mask4 is True
print("Found", len(coords), "bright pixels (mask4)")

if len(coords) > 0:
    r, c = coords[5]  # first bright pixel
    print("First bright pixel at (row, col):", r, c)

    h, w = mask4.shape
    r0 = max(0, r - pixel_radius)
    r1 = min(h, r + pixel_radius + 1)
    c0 = max(0, c - pixel_radius)
    c1 = min(w, c + pixel_radius + 1)

    patch = mask4[r0:r1, c0:c1]

    plt.figure()
    plt.imshow(patch)
    plt.title(f"Patch around bright pixel ({r},{c})")
    plt.show()
else:
    print("No bright pixels found.")

#print("Median of selected region: ", median)
#print("Standard Deviation of selected region: ", stddev)
#plt.show()
#plt.imshow(GreenImage[600:700, 850:950], origin='lower')
#plt.show()