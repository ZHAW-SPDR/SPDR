from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import numpy as np
import sys

# HYPERPARAMS
DISTANCE_METRIC = 'euclidean'
SPLIT = 0.05

# PARAMS
SAMPLES = 20
CENTERS = 3

# Generate Data
X1, y1 = make_blobs(n_samples=SAMPLES, centers=CENTERS, n_features=2, random_state=42)
X2, y2 = make_blobs(n_samples=SAMPLES, centers=CENTERS, n_features=2, random_state=1337)
X3, y3 = make_blobs(n_samples=SAMPLES, centers=CENTERS, n_features=500, random_state=42)
X4, y4 = make_blobs(n_samples=SAMPLES, centers=CENTERS, n_features=500, random_state=1337)
z1 = []
z2 = []
z3 = []
z4 = []

# Pairwise Sequences
# X = Datapoint, Y = Cluster association, Z = Remapped clusters
datasets = [(X1, y1, z1), (X2, y2, z2), (X3, y3, z3), (X4, y4, z4)]
for (x, y, z) in datasets:
    counter = 0
    sequence = []
    sequence.append(x[0])
    for e1, e2 in zip(x, x[1:]):
        distances = []
        for es in sequence:
            distances.append(pdist([es, e2], metric=DISTANCE_METRIC))

        minimum_cluster_distance = min(distances)
        if minimum_cluster_distance < SPLIT:
            sequence.append(e2)
        else:
            z += [counter for _ in range(len(sequence))]
            sequence = [e2]
            counter += 1
    z += [counter for _ in range(len(sequence))]

# TSNE
X3 = TSNE(n_components=2, init='pca').fit_transform(X3)
X4 = TSNE(n_components=2, init='pca').fit_transform(X4)

# Plot
plt.figure(figsize=(10, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(421)
plt.scatter(X1[:, 0], X1[:, 1], c=y1)
for i in range(len(X1)):
    plt.annotate(i, (X1[i, 0], X1[i, 1]))

plt.subplot(422)
plt.scatter(X2[:, 0], X2[:, 1], c=y2)
for i in range(len(X2)):
    plt.annotate(i, (X2[i, 0], X2[i, 1]))

plt.subplot(423)
plt.scatter(X1[:, 0], X1[:, 1], c=z1)
for i in range(len(X1)):
    plt.annotate(i, (X1[i, 0], X1[i, 1]))

plt.subplot(424)
plt.scatter(X2[:, 0], X2[:, 1], c=z2)
for i in range(len(X2)):
    plt.annotate(i, (X2[i, 0], X2[i, 1]))

plt.subplot(425)
plt.scatter(X3[:, 0], X3[:, 1], c=y3)
for i in range(len(X3)):
    plt.annotate(i, (X3[i, 0], X3[i, 1]))

plt.subplot(426)
plt.scatter(X4[:, 0], X4[:, 1], c=y4)
for i in range(len(X4)):
    plt.annotate(i, (X4[i, 0], X4[i, 1]))

plt.subplot(427)
plt.scatter(X3[:, 0], X3[:, 1], c=z3)
for i in range(len(X1)):
    plt.annotate(i, (X1[i, 0], X1[i, 1]))

plt.subplot(428)
plt.scatter(X4[:, 0], X4[:, 1], c=z4)
for i in range(len(X4)):
    plt.annotate(i, (X4[i, 0], X4[i, 1]))

plt.show()