from spdr.afcluster import SPDR_AffinityClustering
import pickle
import numpy as np

with open('data/pickles/TIMIT_DUMMY-01_01.pkl', 'rb') as pFile:
    data = pickle.load(pFile)

afc = SPDR_AffinityClustering()
afc.load(data)
afc.do_cluster(intervals=5)
labels, sil_score, clusters = afc.get_results()
print("Labels: \n%s\nSilhouette Score: \n\t%f\nClusters: \n%s\n" % (labels, sil_score, clusters))
