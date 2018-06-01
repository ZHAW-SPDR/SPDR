"""
The controller for speaker diarization.
Usage: python controller.py [-h] 

Controller for Speaker diarization
optional arguments:
  -h, --help  show this help message and exit
  
"""
import os
import logging
import numpy as np
import pickle
from pyannote.core import Timeline, Segment

from spdr.metrics import SPDR_Metrics
from spdr.segmenter import segment_wave_files_from_to
from spdr.embedding_extractor import EmbeddingExtractor
from spdr.sequencer import SPDR_Sequencer
from spdr.utils import SPDR_Util, get_filename_with_new_extension, get_filename_without_extension
from spdr.hypothesis_generator import define_hypothesis_for_embeddings
from spdr.clustering import clustering_controller
from spdr.handler import SPDR_RT09_Handler
from spdr.vad import SPDR_GMM_Vad


class Controller:
    """
        Runs the pipeline
    """

    def __init__(self):
        self.config = SPDR_Util.load_config()
        self.vad = SPDR_GMM_Vad()
        self.embedding_extractor = EmbeddingExtractor(segment_path=self.config['data']['out'],
                                                      n_segments=self.config['segment']['cap'],
                                                      network_file=self.config["embeddings"]["network"],
                                                      max_audio_length=int(self.config['segment']['size'] / 10) + 1)

    def run(self, ret=False):
        # check audio specification if needed

        handler = SPDR_RT09_Handler()
        dataset = handler.run()
        dataset = dataset[0] if type(dataset) is list else dataset # this should be removed once a voting system is implemented
        condition = self.config['spkr']['condition']
        mdm_warning = """MDM relies on some sort of voting system for all the microphone recordings to perform well.\n
            This feature is not yet implemented and the MDM condition will currently yield bad results.\n
            You might want to use SDM condition instead!"""
        print(mdm_warning if condition.lower() == 'mdm' else '')
        print("SPEAKER COUNT: %d" % len(dataset["speakers"]))

        file_to_progress = (dataset["files"][0])

        if self.config["data"]["use_normalized_audio"]:
            file_to_progress = file_to_progress.replace(".wav", "_normalized.wav")
            if not os.path.isfile(file_to_progress):
                raise Exception("Normalized audio file %s not found. Please run the normalize_audio.sh first.")

        aligned_timeline_in_ms = Controller._align_timeline(self.config['segment']['size'],
                                                            (dataset["start"], dataset["end"]))

        embedding_pickle = os.path.join(self.config["data"]["pickle"], get_filename_with_new_extension(file_to_progress, ".pkl"))

        if os.path.isfile(embedding_pickle):
            with open(embedding_pickle, 'rb') as pFile:
                embeddings = pickle.load(pFile)
        else:
            # split in sequences
            segment_wave_files_from_to(files=[file_to_progress], segment_path=self.config['data']['out'],
                                       segment_size=self.config['segment']['size'],
                                       timeline=aligned_timeline_in_ms)

            # get embedding vector - use https://github.com/stdm/ZHAW_deep_voice/
            embeddings = self.embedding_extractor.extract_embeddings([file_to_progress])

            with open(embedding_pickle, 'wb') as pFile:
                pickle.dump(embeddings, pFile)

        # join similar sequences (i.e. part A of sentence 1 with part B of sentence 1)
        sequencer = SPDR_Sequencer(embeddings=embeddings)
        changepoint_vector = sequencer.find_changepoints()

        print("Number of changepoints: %d" % changepoint_vector.count(0))

        # pass into clustering
        clustering = clustering_controller.ClusteringController(self.config)
        cluster_result = clustering.cluster_embeddings(embeddings, np.array(range(0, len(embeddings))),
                                                        changepoint_vector=changepoint_vector)

        print("Number of clusters: %d" % (max(cluster_result) + 1))

        # run VAD
        segment_folder = os.path.join(self.config['data']['out'],
                                      get_filename_without_extension(os.path.basename(file_to_progress)))

        self.vad.classify_clusters(segment_folder, cluster_result, self.config['vad']['aggressiveness'])

        # get metrics
        uem = Timeline([Segment(dataset["start"], dataset["end"])])
        hypothesis = define_hypothesis_for_embeddings(cluster_result, aligned_timeline_in_ms[0] / 1000)

        # do cleanup
        handler.do_cleanup()
        
        if not ret:
            spdr_metrics = SPDR_Metrics(dataset["reference"], hypothesis, uem)
            metrics = spdr_metrics.get_DiarizationErrorRate(detailed=True)
            print('='*15,'Results','='*15)
            print('DER:\t\t\t%.4f'%metrics['diarization error rate'])
            print('Confusion:\t\t%.4f'%metrics['confusion'])
            print('False Alarm:\t\t%.4f'%metrics['false alarm'])
            print('Correct:\t\t%.4f'%metrics['correct'])
            print('Missed Detection:\t%.4f'%metrics['missed detection'])
            print('Total:\t\t\t%.4f'%metrics['total'])
            print('='*39)

            der = spdr_metrics.get_DiarizationErrorRate(detailed=False)
            purity = spdr_metrics.get_DiarizationPurity(detailed=False)
            coverage = spdr_metrics.get_DiarizationCoverage(detailed=False)
            cpfscore = spdr_metrics.get_DiarizationCoveragePurityFScore(detailed=False)
            confusion = metrics['confusion']
            falsealarm = metrics['false alarm']
            correct = metrics['correct']
            misseddetection = metrics['missed detection']
            ptext = "DER: %.4f | Purity: %.4f | Coverage: %.4f | PC-F-Score: %.4f \nConfusion: %.2f | False Alarm: %.2f | Correct: %.2f | Missed Detection %.2f" % (\
                der, \
                purity, \
                coverage, \
                cpfscore, \
                confusion, \
                falsealarm, \
                correct, \
                misseddetection
                )
            spdr_metrics.get_Plot(title=dataset['id'], text=ptext)
        else:
            return {
                'dataset': dataset,
                'reference': dataset["reference"],
                'hypothesis': hypothesis,
                'uem': uem,
                'cluster_result': cluster_result,
                'aligned_timeline_in_ms': aligned_timeline_in_ms[0] / 1000
            }

    @staticmethod
    def _align_to_duration(duration, time, is_start):
        if time % duration == 0:
            return time

        if is_start:
            return time - (time % duration)
        return time + (duration - (time % duration))

    @staticmethod
    def _align_timeline(duration, timeline):
        return (Controller._align_to_duration(duration, int(timeline[0] * 1000), is_start=True),
                Controller._align_to_duration(duration, int(timeline[1] * 1000), is_start=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    controller = Controller()
    controller.run()
