import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ['EDI_20071128-1000', 'EDI_20071128-1500', 'IDI_20090128-1600', 'IDI_20090129-1000', 'NIST_20080201-1405', 'NIST_20080227-1501', 'NIST_20080307-0955']

def split_list(L, n):
    for i in range(0, len(L), n):
        yield L[i:i+n]

result_pickle = os.path.join('./data/pickles/randomresults.pkl')

with open(result_pickle, 'rb') as pFile:
    results = pickle.load(pFile)

experiment_results = list(split_list(results, 10))
for idx, er in enumerate(experiment_results):
    print('Meeting', DATASETS[idx])
    runs = []
    data = [] 
    for run, res in enumerate(er):
        fscore = 2 * (res[6] * res[7])/(res[6] + res[7])
        print('\t', 'Run #%d' % (run), '-', 'DER: %.4f | Purity: %.4f | Coverage: %.4f | F-Score: %.4f' %(res[0], res[6], res[7], fscore))
        runs.append(run+1)
        data.append([res[0], res[6], res[7], fscore])

    ders = [x[0] for x in data]
    mean_der = [np.mean(ders)]*len(ders)
    plt.figure(figsize=(10,6))
    plt.plot(runs,data)
    plt.title('Random Diarization Performance for Meeting %s' % DATASETS[idx])
    plt.xlabel('Runs')
    plt.xticks(runs)
    plt.ylabel('Metric Value')
    plt.plot(runs, mean_der, linestyle='--')
    plt.yticks(np.linspace(0,1.1,12))
    plt.legend(('DER','Purity','Coverage', 'F-Score','Mean DER %.4f' % mean_der[0]), loc='best')
    plt.grid()
    plt.savefig("./data/plots/experiment5_random/random_%s.png" % (DATASETS[idx]))
    plt.close()