#import libraries and files
from GONet_Wizard.GONet_utils import GONetFile
import matplotlib.pyplot as plt
import numpy as np

#assign image that we want from GONetFile
go = GONetFile.from_file(r"C:\Users\spall\Desktop\GONet\Testing Images\202_250628_063009_1751092241.tiff")

#create an empty figure 
fig = plt.figure()

#add a subplot "ax" at the first position in a 1x1 grid 
ax = fig.add_subplot(111)

#add subplots at x position on my 1x3 grid
#ax = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

##of the green channel get the full length of the row (all rows) and every column after index 1000
row = go.green[0:len(go.green[1]), 10:20]

#print the average of the row
print(row.mean())
##print row attributes 
print(type(row), len(row), row.shape)

#subtract the mean from the entire image
GreenImage = go.green

GreenImage = GreenImage - row.mean()

print(go.green.mean())

print(GreenImage.mean())

go.remove_overscan()

print(go.green.mean())
##plot the row specified 
plt.imshow(GreenImage)
plt.show()
#exit()

#print(row)

##create numpy list: index 0 of row, going to the specified length of green channel, with step size of one int. 
x = np.arange(0, len(go.green[500, :]), 1)
#print(type(x))

#fit a line using our calculated x (x) calculated y(row) that is linear (1) y =mx+b
pp = np.polyfit(x, row, 1)

print(pp)

#get the mean number of pixel value across the entire green channel and the std
m = np.mean(go.green)
stddev = np.std(go.green)

print(stddev)

#compare the average to the y intercept (pp[1])
print(m, pp[1])
print(np.abs(m - pp[1]))
y = pp[0] * x + pp[1]
plt.plot(x, row)
plt.plot([x[0], x[-1]], [y[0], y[-1]])
#plt.plot(x, y)

plt.show()

#print(pp)
#exit()
ax.plot(row)

##find min/max manually 
min = row[1]
max = row[1]
for i in row:
    if(i < min):
        min = i
    if(i > max):
        max = i

print(min , max)

diff = max - min

print(diff)





#plt.show()
#to subtract "plateu" from overall 
#go.remove_overscan()

#for i,channel in enumerate(['blue', 'green', 'red']):
#    ax[i].imshow(go.get_channel(channel))

#ax.imshow(go.green)

#show hist 
#ax.hist(go.green)

#plt.show()