import os
import pickle

RT09FOLDER = './data/in/RT09'
RUNFOLDER = './data/runs'
GROUNDTRUTH = dict()
CLUSTER_RESULTS_Vox = dict()
CLUSTER_RESULTS_Timit = dict()
for path, dirs, files in os.walk(RT09FOLDER):
    if path == RT09FOLDER:
        continue
    MEETING = path.replace(RT09FOLDER, '').replace('/','')
    speakers_ = []
    for f in files:
        if f.endswith('.txt'):
            FILEHANDLE = os.path.join(path, f)
            lines = [line.strip() for line in open(FILEHANDLE, 'r').readlines()]
            for line in lines:
                # check whether tab or space delimited
                entry = line.split(' ')
                if len(entry) < 3:
                    entry = line.split('\t')
                ref_spkr = (entry[2]).replace(':', '')
                speakers_.append(ref_spkr)
    speakers = set(speakers_)
    GROUNDTRUTH[MEETING.lower()] = speakers 

meetings = [k for k, _ in GROUNDTRUTH.items()]

for path, dirs, files in os.walk(RUNFOLDER):
    for m in meetings:
        if m in path:
            MEETING = m
            NETWORK = path.split(".MODEL_",1)[1]
            PICKLE = os.path.join(path, 'result.pkl')
            with open(PICKLE, 'rb') as pFile:
                results = pickle.load(pFile)
                if NETWORK == 'pairwise_lstm_1251_voxceleb_best':
                    CLUSTER_RESULTS_Vox[MEETING] = results['cluster_result']
                else:
                    CLUSTER_RESULTS_Timit[MEETING] = results['cluster_result']


for k, v in GROUNDTRUTH.items():
    print('='*40)
    print('Meeting', k.upper())
    print(v)
    print('Count:\t\t', len(v))
    print('Predicted:')
    print('Timit',':\t\t', len(set(CLUSTER_RESULTS_Timit[k.lower()]))-1)
    print('Vox',':\t\t', len(set(CLUSTER_RESULTS_Vox[k.lower()]))-1)