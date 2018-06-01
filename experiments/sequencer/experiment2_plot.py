import pickle
import os
import sys
import random
import numpy as np
from scipy.spatial.distance import pdist
from scipy.interpolate import *
import matplotlib.pyplot as plt
import math

np.random.seed(1337)
random.seed(1337)
SCALE_NOTSAME = False
SCALE_SAME = True
PICKLEFOLDER = 'data/pickles/'
#PICKLE_NAME_STEM = 'TIMIT_R1_'
#PICKLE_NAME_STEM = 'RT09_EDI_17_PHRASES_4_SPEAKERS_TIMIT_LSTM_01_'
PICKLE_NAME_STEM = 'RT09_EDI_17_PHRASES_4_SPEAKERS_VOXCELEB_LSTM_01_'
DSPICKLE = PICKLE_NAME_STEM+'dataset.pkl'
PAIRPICKLE = PICKLE_NAME_STEM+'prodpairs.pkl'
DIST_MET = 'cosine'

dataset_pickle = os.path.join(PICKLEFOLDER, DSPICKLE)
with open(dataset_pickle, 'rb') as pFile:
    dataset = pickle.load(pFile)

#tmp_data = list([x for x in dataset[::12]])
tmp_data = list([x for x in dataset[::]])
random.shuffle(tmp_data)
dataset = tmp_data
pair_pickle = os.path.join(PICKLEFOLDER, PAIRPICKLE)
if not os.path.isfile(pair_pickle):
    data = [(x, y) for x in dataset[:int(len(dataset)/2)] for y in dataset[int(len(dataset)/2):]] # join every item with every item
    with open(pair_pickle, 'wb') as pFile:
        pickle.dump(data, pFile)
else:
    with open(pair_pickle, 'rb') as pFile:
        data = pickle.load(pFile)
    
    X = []
    Y = []
    for (ds0, ds1) in data:
        same_speaker = 1 if ds0[1] == ds1[1] else 0
        if np.random.randint(low=0, high=8) == 1 or same_speaker == 1:
            X.append(pdist([ds0[0], ds1[0]], metric=DIST_MET))
            Y.append(same_speaker)

    print('0:', Y.count(0), Y.count(0)/len(Y))
    print('1:', Y.count(1), Y.count(1)/len(Y))
    print('Max:', max(X))
    print('Min:', min(X))
    
    zeros = [i for i, x in enumerate(Y) if x == 0]
    if SCALE_NOTSAME:
        zeros = list([x for x in zeros[::15]])
    else:
        zeros = list([x for x in zeros[::]])
    ones = [i for i, x in enumerate(Y) if x == 1]
    if SCALE_SAME:
        ones = list([x for x in ones[::5]])
    else:
        ones = list([x for x in ones[::]])
    print('0:', len(zeros))
    print('1:', len(ones))
    dzeros = np.array(X)[zeros]
    dones = np.array(X)[ones]
    data = [dzeros, dones]
    print('Max 0:', max(dzeros))
    print('Min 0:', min(dzeros))
    print('Max 1:', max(dones))
    print('Min 1:', min(dones))
    
    plt.figure(figsize=(9,6))
    plt.scatter(len(dzeros) * [0], dzeros, marker='.', label='Not same')
    plt.scatter(len(dones) * [1], dones, marker='.', label='Same')
    plt.xlim(-0.5,1.5)
    plt.ylim(0,1.5)
    plt.title('Distances')
    plt.ylabel('Distance')
    plt.legend()
    plt.savefig("./data/plots/experiment2_cutoff/%s%s_cutoff_distance.png" % (PICKLE_NAME_STEM, DIST_MET))
    plt.close()

    plt.figure(figsize=(9,6))
    
    plt.boxplot(data, labels=['Not same', 'Same'])
    plt.ylabel('Distance')

    plt.title('Distances Box Plot')
    plt.ylim(0,1.5)
    plt.savefig("./data/plots/experiment2_cutoff/%s%s_cutoff_distance_boxplot.png"  % (PICKLE_NAME_STEM, DIST_MET))
    plt.close()
    plt.figure(figsize=(9,6))
    plt.title('Distribution')
    unique_zeros, counts_zeros = np.unique([round(x[0],2) for x in data[0]], return_counts=True)
    unique_ones, counts_ones = np.unique([round(x[0],2) for x in data[1]], return_counts=True)
    plt.fill_between(unique_zeros, counts_zeros, color='b', alpha=.5, interpolate=False)
    plt.fill_between(unique_ones, counts_ones, color='g', alpha=.5, interpolate=False)
    plt.legend(['Not same', 'Same'])
    plt.xlabel('Distance')
    plt.ylabel('Count')

    plt.xticks(np.linspace(0,1,21))
    plt.xlim(0,1)
    plt.ylim(0,max(max(counts_zeros), max(counts_ones)))
    plt.savefig("./data/plots/experiment2_cutoff/%s%s_cutoff_distribution.png" % (PICKLE_NAME_STEM, DIST_MET))
    plt.close()
    
    plt.figure(figsize=(9,6))
    
    xs = np.linspace(0,1,300)

    poly_deg = 10
    coefs_zeros = np.polyfit(unique_zeros, counts_zeros, poly_deg)
    coefs_ones = np.polyfit(unique_ones, counts_ones, poly_deg)
    ys_zeros = np.polyval(coefs_zeros, xs)
    ys_ones = np.polyval(coefs_ones, xs)
    
    # fix poly problems
    neg_zeros = 0 if [i for i,x in enumerate(ys_zeros) if x<0] == [] else [i for i,x in enumerate(ys_zeros) if x<0][0]
    neg_ones = 0 if [i for i,x in enumerate(ys_ones) if x<0] == [] else [i for i,x in enumerate(ys_ones) if x<0][0]
    ys_zeros = [0 if i <= neg_zeros else x for i, x in enumerate(ys_zeros)]
    ys_ones = [0 if i <= neg_ones else x for i, x in enumerate(ys_ones)]

    plt.fill_between(xs, ys_zeros, color='r', alpha=.5, interpolate=False)
    plt.fill_between(xs, ys_ones, color='y', alpha=.5, interpolate=False)
    plt.legend(['Not same', 'Same'])
    plt.xlabel('Distance')

    plt.title('Distribution Smoothed')
    plt.xticks(np.linspace(0,1,21))
    plt.ylabel('Count')
    plt.xlim(0,1)
    plt.ylim(0,max(max(counts_zeros), max(counts_ones)))
    plt.savefig("./data/plots/experiment2_cutoff/%s%s_cutoff_distribution_smoothed.png" % (PICKLE_NAME_STEM, DIST_MET))
    plt.close()
    