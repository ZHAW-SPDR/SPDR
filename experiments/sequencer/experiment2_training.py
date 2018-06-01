import pickle
import os
import sys
import random
import numpy as np
from scipy.spatial.distance import pdist
from sklearn import svm, preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

random.seed(1337)

PICKLEFOLDER = 'data/pickles/'
PICKLE_NAME_STEM = 'TIMIT_R1_'
PICKLE = PICKLE_NAME_STEM+'dataset.pkl'
DIST_MET = 'cosine'

dataset_pickle = os.path.join(PICKLEFOLDER, PICKLE)
with open(dataset_pickle, 'rb') as pFile:
    dataset = pickle.load(pFile)

equal_pairs = [e for e in zip(dataset, dataset)]
dataset_shuffled = list(dataset)
random.shuffle(dataset_shuffled)
random_pairs = [e for e in zip(dataset, dataset_shuffled)]

data = equal_pairs + random_pairs
X = []
Y = []
for (ds0, ds1) in data:
    X.append(pdist([ds0[0], ds1[0]], metric=DIST_MET))
    Y.append(1 if ds0[1] == ds1[1] else 0)

print('0:', Y.count(0), Y.count(0)/len(Y))
print('1:', Y.count(1), Y.count(1)/len(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC())

clf.fit(X_train, Y_train)

#y_pred = clf.predict(X_test)
#print(classification_report(Y_test, y_pred), file=sys.stderr)
y_pred = cross_val_predict(clf, X, Y, cv=100)

print(classification_report(Y, y_pred), file=sys.stderr)


model_pickle = os.path.join(PICKLEFOLDER, PICKLE_NAME_STEM+'model.pkl')
joblib.dump(clf, model_pickle)