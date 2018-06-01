import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

np.random.seed(0)
f, axs = plt.subplots(nrows=((20 + 4) // 4), ncols=4)
f.set_figwidth(15)
f.set_figheight(15)
plt.setp(axs, xlim=[0,10], ylim=[0,10], xticks=np.linspace(0,10,11), yticks=np.linspace(0,10,11))

cluster_centers = [[3,6], [7,3]]
CUTOFF = 3.
X = [3,7]
Y = [6,3]
colors = [1,2]
index = 0
for row in axs:
    for elem in row:
        x = np.random.randint(low=1, high=10)
        y = np.random.randint(low=1, high=10)
        cval = 0
        cval = 1 if pdist([np.array(cluster_centers[0]), np.array([x,y])], metric='euclidean') <= CUTOFF else cval
        cval = 2 if pdist([np.array(cluster_centers[1]), np.array([x,y])], metric='euclidean') <= CUTOFF else cval
        colors.append(cval)
        X.append(x)
        Y.append(y)
        elem.scatter(X,Y, c=np.array(colors))
        index += 1
plt.tight_layout()
plt.savefig("./data/plots/experiment2_linkage/plot.png")
plt.close()