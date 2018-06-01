import pickle
from sklearn.externals import joblib
import os
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import numpy as np

np.random.seed(1337)

PICKLEFOLDER = 'data/pickles/'
PICKLE_NAME_STEM = 'TIMIT_R1_'
PICKLE = PICKLE_NAME_STEM+'dataset.pkl'
model_pickle = os.path.join(PICKLEFOLDER, PICKLE_NAME_STEM+'model.pkl')

PAIRPICKLE = PICKLE_NAME_STEM+'prodpairs.pkl'
DIST_MET = 'cosine'

pair_pickle = os.path.join(PICKLEFOLDER, PAIRPICKLE)
with open(pair_pickle, 'rb') as pFile:
    data = pickle.load(pFile)
X = []
Y = []
Z = []
predicitions = []
for (ds0, ds1) in data:
    same_speaker = 1 if ds0[1] == ds1[1] else 0
    if np.random.randint(low=0, high=8) == 1 or same_speaker == 1:
        X.append(pdist([ds0[0], ds1[0]], metric=DIST_MET))
        Y.append(same_speaker)

clf = joblib.load(model_pickle)

datasets = [(X,Y,Z)]
Z = clf.predict(X)

print('0:', np.count_nonzero(Z == 0), np.count_nonzero(Z == 0)/len(Y))
print('1:', np.count_nonzero(Z == 1), np.count_nonzero(Z == 1)/len(Y))
print('Max:', max(X))
print('Min:', min(X))
plt.figure(figsize=(20,8))
zeros = [i for i, x in enumerate(Z) if x == 0]
ones = [i for i, x in enumerate(Z) if x == 1]
dzeros = np.array(X)[zeros]
dones = np.array(X)[ones]
data = [dzeros, dones]
print('Max 0:', max(dzeros))
print('Min 0:', min(dzeros))
print('Max 1:', max(dones))
print('Min 1:', min(dones))

plt.subplot(121)
plt.scatter(len(dzeros) * [0], dzeros, marker='.', label='Not same')
plt.scatter(len(dones) * [1], dones, marker='.', label='Same')
plt.xlim(-0.5,1.5)
plt.ylim(0.1,1.5)
plt.legend()
plt.subplot(122)
plt.boxplot(data, labels=['Not same', 'Same'])
plt.ylim(0.1,1.5)
plt.tight_layout()
plt.show()
# for (x, y, z) in datasets:
#     counter = 0
#     sequence = []
#     sequence.append(0)
#     for e in x[1:]:
#         prediction = clf.predict([e])
#         predicitions.append(prediction)
#         if prediction[0] == 1:
#             sequence.append(e)
#         else:
#             z += [counter for _ in range(len(sequence))]
#             sequence = [e]
#             counter += 1
#     z += [counter for _ in range(len(sequence))]

# A = [x[0] for x in dataset]
# B = [x[1] for x in dataset]
# le = LabelEncoder()
# le.fit(list(set(B)))
# A_TSNE = TSNE(n_components=2, init='pca').fit_transform(A)
# plt.scatter(A_TSNE[:, 0], A_TSNE[:, 1], c=le.transform(B))
# plt.show()