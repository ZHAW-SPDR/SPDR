import numpy as np
import os
import pickle
from spdr.handler import SPDR_RT09_Handler
from scipy.spatial.distance import pdist
from pyannote.core import Annotation, Segment
from pyannote.metrics.segmentation import *
import matplotlib.pyplot as plt

# values
from spdr.utils import get_filename_with_new_extension

metric_data = []
dsID = 'EDI_20071128-1000'

# parameters
PRINT = False
DISTANCE_METRIC = 'cosine'

# load embeddings and dataset
handler = SPDR_RT09_Handler()
dataset = handler.run()[0]
embedding_pickle = os.path.join('./data/pickles/', get_filename_with_new_extension(dataset["files"][0], ".pkl"))
with open(embedding_pickle, 'rb') as pFile:
    embeddings = pickle.load(pFile)

# prepare
pairs = [e for e in zip(embeddings, embeddings[1:])]
cutoffvalues = np.linspace(0,1,21)
for CUTOFF in cutoffvalues:

    # processing
    similarity = [0]
    distances = [0]

    for (p0, p1) in pairs:
        distance = pdist([p0, p1], metric='cosine')
        distances.append(distance)
        similarity.append(0 if distance > CUTOFF else 1)

    # change point detector
    previous = None
    changepoints = []
    for i, e in enumerate(similarity):
        if i == 0:
            previous = e
            #print("Speaker Change detected at 0")
            changepoints.append(dataset['start'])
        if previous == 1 and e == 0:
            #print("Speaker Change detected at %d" % (i*500))
            changepoints.append(dataset['start'] + (i * 0.5))
        previous = e

    # reference and hypothesis generation
    ref = dataset['reference']
    hyp = Annotation()
    prev = changepoints[0]
    for now in changepoints[1:]:
        hyp[Segment(prev, now)] = ''
        prev = now
    hyp[Segment(prev, dataset['end'])] = ''

    # print
    m_seg_pre = SegmentationPrecision()
    m_seg_pre_val = m_seg_pre.compute_metric(m_seg_pre.compute_components(ref, hyp))
    m_seg_rec = SegmentationRecall()
    m_seg_rec_val = m_seg_rec.compute_metric(m_seg_rec.compute_components(ref, hyp))
    m_seg_cov = SegmentationCoverage()
    m_seg_cov_val = m_seg_cov.compute_metric(m_seg_cov.compute_components(ref, hyp))
    m_seg_pur = SegmentationPurity()
    m_seg_pur_val = m_seg_pur.compute_metric(m_seg_pur.compute_components(ref, hyp))
    try:
        f1 = 2*((m_seg_pre_val*m_seg_rec_val)/(m_seg_pre_val+m_seg_rec_val))
    except ZeroDivisionError:
        f1 = 0.

    if PRINT:
        print('Results with Cutoff %.2f:' % CUTOFF)
        print('Segmentation Precision: \t %.8f'% m_seg_pre_val)
        print('Segmentation Recall: \t\t %.8f'% m_seg_rec_val)
        print('Segmentation Coverage: \t\t %.8f'% m_seg_cov_val)
        print('Segmentation Purity: \t\t %.8f'% m_seg_pur_val)
    else:
        metric_data.append([m_seg_pre_val, m_seg_rec_val, m_seg_cov_val, m_seg_pur_val, f1])
    
# plot
plt.figure(figsize=(10,6))
plt.plot(cutoffvalues,metric_data)
plt.title('Grid Search for Cutoff Value')
plt.xlabel('Cutoff Value')
plt.xticks(cutoffvalues)
plt.ylabel('Metric Value')
plt.yticks(np.linspace(0,1,11))
plt.legend(('Precision','Recall','Coverage','Purity','F1-Score'), loc='best')
plt.grid()
plt.savefig("./data/plots/experiment2_gridsearch/%s.png" % (dsID))
plt.close()
plt.figure(figsize=(12,8))
coverage = [m[2] for m in metric_data]
purity = [m[3] for m in metric_data]
b1 = plt.bar(cutoffvalues, coverage, 0.35)
b2 = plt.bar(cutoffvalues, purity, 0.35, bottom=coverage)
plt.title('Grid Search for Cutoff Value')
plt.xlabel('Cutoff Value')
plt.xticks(cutoffvalues)
plt.ylabel('Stacked Metric Value')
plt.yticks(np.linspace(0,2,41))
plt.legend((b1[0], b2[0]),('Coverage','Purity'), loc='best')
plt.grid()
plt.savefig("./data/plots/experiment2_gridsearch/stacked_%s.png" % (dsID))
plt.close()
