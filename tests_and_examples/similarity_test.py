from sklearn.metrics import pairwise_distances
import numpy as np
from array import array

A = np.array([[1,2,3,4]])
B = np.array([[5,6,7,8]])
C = np.array([[3,-1,7,9]])
D = np.array([[4,3,2,1]])
METRICS = ['euclidean', 'manhattan', 'cosine']
MATCHUPS = [[A,A],[A,B],[A,C],[A,D],[B,B],[B,C],[B,D],[C,C],[C,D],[D,D]]

for metric in METRICS:
    for match in MATCHUPS:
        print('Metric %s using (%s) and (%s)' % (metric, match[0], match[1]))
        D = pairwise_distances(match[0], match[1])
        gamma = 1 / len(D)
        S1 = np.exp(-D * gamma)
        S2 = 1. / (D / np.max(D))
        print(D)
        print(S1)
        print(S2)