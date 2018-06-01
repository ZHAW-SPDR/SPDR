import numpy as np
import os
import pickle
from spdr.handler import SPDR_RT09_Handler
from spdr.segmenter import segment_wave_files_from_to
from spdr.utils import SPDR_Util, get_filename_with_new_extension
from spdr.embedding_extractor import EmbeddingExtractor
from controller import Controller
from spdr.metrics import SPDR_Metrics
from spdr.hypothesis_generator import define_hypothesis_for_embeddings
from pyannote.core import Timeline, Segment

SEEDS = [30938, 29772, 24801, 31810, 44803, 46861, 42649, 33132, 25087, 18062]
#SEEDS = [30938]
DATASETS = ['EDI_20071128-1000', 'EDI_20071128-1500', 'IDI_20090128-1600', 'IDI_20090129-1000', 'NIST_20080201-1405', 'NIST_20080227-1501', 'NIST_20080307-0955']
#DATASETS = ['EDI_20071128-1000']

config = SPDR_Util.load_config()
results = []

for i, seed in enumerate(SEEDS):
    print('Running Experiment %d with seed %d' %(i+1, seed))
    np.random.seed(seed)
    for j, dataset in enumerate(DATASETS):
        handler = SPDR_RT09_Handler(dataset)
        dataset = handler.run()
        dataset = dataset[0]
        file_to_progress = (dataset["files"][0])
        embedding_pickle = os.path.join('./data/pickles/', get_filename_with_new_extension(file_to_progress, ".pkl"))
        if os.path.isfile(embedding_pickle):
            with open(embedding_pickle, 'rb') as pFile:
                embeddings = pickle.load(pFile)
        else:
            timeline = Controller._align_timeline(500, (float(dataset["start"]),
                                                                               float(dataset["end"])))

            segment_wave_files_from_to(files=[file_to_progress], segment_path='./data/out/',
                                       segment_size=500,
                                       timeline=timeline)

            embedding_extractor = EmbeddingExtractor(segment_path='./data/out/',
                                                      n_segments=10000,
                                                      network_file="pairwise_lstm_100_best.h5",
                                                      max_audio_length=int(500 / 10) + 1)
            embeddings = embedding_extractor.extract_embeddings([file_to_progress])

            with open(embedding_pickle, 'wb') as pFile:
                pickle.dump(embeddings, pFile)

        # Random Stuff
        cluster_result = []
        clusters = np.random.randint(low=2, high=20)
        while len(cluster_result) < len(embeddings):
            cluster_result.append(np.random.randint(low=0, high=clusters))

        if config['hypothesis']['scaledown']:
            uem = Timeline([Segment(float(dataset["start"]) / 1000, float(dataset["end"]) / 1000)])
        else:
            uem = Timeline([Segment(float(dataset["start"]), float(dataset["end"]))])

        non_speech = np.random.choice(list(set(cluster_result)))
        #non_speech = list(set(np.random.choice(cluster_result, np.random.randint(0,len(cluster_result)))))
        
        # note: for a re run, this might need to be reworked, because the current API does not allow to specify a non_speech list
        hypothesis = define_hypothesis_for_embeddings(cluster_result, non_speech)

        spdr_metrics = SPDR_Metrics(dataset["reference"], hypothesis, uem)
        metrics = spdr_metrics.get_DiarizationErrorRate(detailed=True)
        print('='*15,'Results','='*15)
        print('='*10, dataset['id'],'='*10)
        print('DER:\t\t\t%.4f'%metrics['diarization error rate'])
        print('Confusion:\t\t%.4f'%metrics['confusion'])
        print('False Alarm:\t\t%.4f'%metrics['false alarm'])
        print('Correct:\t\t%.4f'%metrics['correct'])
        print('Missed Detection:\t%.4f'%metrics['missed detection'])
        print('Total:\t\t\t%.4f'%metrics['total'])
        print('Purity:\t\t\t%.4f'% spdr_metrics.get_DiarizationPurity())
        print('Coverage:\t\t%.4f'% spdr_metrics.get_DiarizationCoverage())
        print('='*39)
        results.append([metrics['diarization error rate'], metrics['confusion'], metrics['false alarm'], metrics['correct'], metrics['missed detection'], metrics['total'], spdr_metrics.get_DiarizationPurity(), spdr_metrics.get_DiarizationCoverage()])

results_pickle = os.path.join('./data/pickles/', "randomresults.pkl")
with open(results_pickle, 'wb') as pFile:
    pickle.dump(results, pFile)