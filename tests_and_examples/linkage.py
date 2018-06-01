import pickle
import os
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)
cutoff = 0.3
linkage = 'single'

PICKLEFOLDER = 'data/pickles/'
PICKLE_NAME_STEM = 'TIMIT_DUMMY-01_01'
emb_pickle = os.path.join(PICKLEFOLDER, PICKLE_NAME_STEM+'.pkl')
with open(emb_pickle, 'rb') as pFile:
    embeddings = pickle.load(pFile)

similarity = [0]
seen_embeddings = [embeddings[0]]

for embedding in embeddings:
    distances = []
    for seen_embedding in seen_embeddings:
        distances.append(pdist([seen_embedding, embedding], metric='cosine'))
    
    best_dist = float("inf")
    if linkage == 'single':
        best_dist = min(distances)
    elif linkage == 'complete':
        best_dist = max(distances)
    elif linkage == 'last':
        best_dist = distances[-1]
    else:
        raise Exception("Not supported linkage")

    vec_elem = 0 if best_dist > cutoff else 1
    if vec_elem == 0:
        print("Speaker change detected - %f" % best_dist)
        seen_embeddings = [embedding]
    else:
        print("Still same speaker")
        seen_embeddings.append(embedding)
    similarity.append(vec_elem)

print(similarity)
