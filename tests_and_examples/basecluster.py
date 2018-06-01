import numpy as np
import pickle
from sklearn.metrics import pairwise as pw
from scipy.spatial.distance import pdist, cdist

with open('data/pickles/TIMIT_DUMMY-01_01.pkl', 'rb') as pFile:
    data = pickle.load(pFile)
CUTOFF = 0.1
np.random.seed(56676857)
#data = np.random.randint(low=0, high=10, size=(20,2))
bkp = np.copy(data)
current_datapoint = -1
clusters = []
current_sequence = []

max_dist = max(pdist(data, metric='cosine'))
#data /= max_dist

for idx in range(0, len(data)+1):
    if idx == len(data):
        # cleanup
        clusters.append(current_sequence)
        mean = np.mean(data[current_sequence], axis=0)
        for i in range(0, len(current_sequence)):
            data[current_sequence] = mean
        continue
    label = data[idx]
    if idx == 0:
        current_datapoint = label
        current_sequence.append(idx)
        continue    
    distance = pdist([current_datapoint, label], metric='cosine')
    if distance/max_dist > CUTOFF:
        # start new sequence
        clusters.append(current_sequence)
        mean = np.mean(data[current_sequence], axis=0)
        for i in range(0, len(current_sequence)):
            data[current_sequence] = mean
        current_sequence = []
        current_sequence.append(idx)
        current_datapoint = label
    else:
        current_sequence.append(idx)

#print(bkp)
#print(data)
print("ID: \tidxs | elems")
for i, _ in enumerate(clusters):
    print("%d: \t %s " % (i, clusters[i]))
    #print("%d: \t %s | %s => %s" % (i, clusters[i], bkp[clusters[i]], data[clusters[i]]))