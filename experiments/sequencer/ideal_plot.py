import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

plt.figure(figsize=(9,6))
plt.title('Ideal Distribution')

mean = 0; 
std = 1; 
variance = np.square(std)
x1 = np.arange(0,np.pi,.01)
x2 = np.arange(np.pi-.6,2*np.pi-.6,.01)

f1 = np.square(np.sin(x1))/2
f2 = np.square(np.sin(x2+.6))/1.5
plt.fill_between(x2, f2, color='b', alpha=.5, interpolate=False)
plt.fill_between(x1, f1, color='g', alpha=.5, interpolate=False)
#plt.fill_between(x1, -np.cos(x1), color='b', alpha=.5, interpolate=False)
#plt.fill_between(x2, -np.cos(x2), color='g', alpha=.5, interpolate=False)
plt.legend(['Not same', 'Same'])

plt.xlabel('Distance')
plt.ylabel('Count')
plt.ylim(0,1)
#plt.xticks(np.linspace(0,1,21))
plt.axes().set_xticklabels([])
plt.axes().get_xaxis().set_ticks([])
plt.axes().set_yticklabels([])
plt.axes().get_yaxis().set_ticks([])
plt.savefig("./data/plots/experiment2_cutoff/ideal_cutoff_distribution.png")
plt.close()

plt.figure(figsize=(9,6))
    
plt.boxplot([x2,x1], labels=['Not same', 'Same'])
plt.ylabel('Distance')

plt.title('Distances Box Plot')
#plt.ylim(0,1.5)
plt.axes().get_yaxis().set_ticks([])
plt.savefig("./data/plots/experiment2_cutoff/ideal_cutoff_distance_boxplot.png" )
plt.close()