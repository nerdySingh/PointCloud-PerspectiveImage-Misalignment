import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = open('cam_coords_camcoord.fuse', 'rb')
points = f.readlines()
f.close()

pnts = np.array([np.array(points[i].strip().split(' ')[:-1]).astype(np.float) for i in xrange(len(points)) if i % 64 == 0])
intensities = np.array([np.array(points[i].strip().split(' ')[-1]).astype(np.float) for i in xrange(len(points)) if i % 64 == 0])
intensities /= intensities.max()
intensities.astype(np.uint)

y = np.zeros([intensities.shape[0], 3])
y[:, 0] = intensities
y[:, 1] = intensities
y[:, 2] = intensities

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pnts[:, 0], pnts[:, 1], pnts[:, 2], c=y)
ax.scatter(0, 0, 0, c=(.1, .5, .2))
#plots = []
#color = np.zeros(3)
#cidx = 0
#for sz in sizes:
#  grph = []
#  for c in res[sz].iterkeys():
#      times = res[sz][c]['fraction']
#      if times[0] > 0.:
#          for i in xrange(times.shape[0]):
#              grph.append([c, i+1, times[i]])
#  grph = np.array(grph)
#  color[cidx%3] += .2
#  tmp = ax.scatter(grph[:, 0], grph[:, 1], grph[:, 2], c=tuple(color))
#  cidx += 1
#  plots.append(tmp)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#ax.text2D(0.05, 0.95, 'Performance Loss Plot', transform=ax.transAxes)
#ax.legend(plots, ['1000x1000', '2000x2000', '3000x3000', '4000x4000', '5000x5000', '6000x6000', '7000x7000', '8000x8000', '9000x9000'])

plt.show()