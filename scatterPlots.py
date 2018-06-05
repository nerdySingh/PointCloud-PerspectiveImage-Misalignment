import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
my_file = open("Cam_coords.fuse","r")
scatt=[]
def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin
n=100



fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
for x in my_file:
	scatt.append(x.strip().split())

for i in range(0,10000):
	ax.scatter(float(scatt[i][0]), float(scatt[i][1]), float(scatt[i][3]), c='r', marker='o')
	print("Value-Plotted:",scatt[i])

    	#xs = scatt[i][0]
    	#ys = scatt[i][1]
    	#zs= scatt[i][3]
#ax.scatter(-83.7212, 97.1149, 135.196, c='r', marker='o')
    #xs = randrange(n, 23, 32)
    #ys = randrange(n, 0, 100)
    #zs = randrange(n, zlow, zhigh)
    #ax.scatter(xs, ys, zs, c=c, marker=m)
#-83.7212 97.1149 135.196 10

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
#print(len(scatt))
