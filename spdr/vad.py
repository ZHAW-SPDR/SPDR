"""
    An API for Voice Activity Detection

    retrieved from https://github.com/wiseman/py-webrtcvad
"""
import os
import pickle
from abc import ABC, abstractmethod

import collections

import librosa
import numpy as np
from .utils import SPDR_Util

ALLOWED_EXTENSIONS = ('wav', 'mp3')


class Vad(ABC):
    @abstractmethod
    def classify_segments(self, segment_path):
        pass

    def classify_clusters(self, segment_path, cluster_result, aggressiveness):
        per_cluster_speech_map = collections.defaultdict(list)
        if aggressiveness == 'high':
            threshold = 0.25
        elif aggressiveness == 'low':
            threshold = 0.75
        else:
            threshold = 0.5

        for i, is_speech in enumerate(self.classify_segments(segment_path)):
            per_cluster_speech_map[cluster_result[i]].append(is_speech)

            if not is_speech:
                cluster_result[i] = -1

        for (cluster, speech_map) in per_cluster_speech_map.items():
            if speech_map.count(False)/len(speech_map) > threshold:
                cluster_result[np.where(cluster_result == cluster)] = -1


class SPDR_GMM_Vad(Vad):
    def __init__(self):
        self.config = SPDR_Util.load_config()
        self.segment_size = self.config["segment"]["size"]
        with open("data/vad_models/vad_voxceleb.pkl", "rb") as model_file:
            self.gmm = pickle.load(model_file)

    @staticmethod
    def _get_mfcc_with_deltas(filename, segment_size_ms):
        duration = librosa.get_duration(filename=filename)
        duration_ms = int(duration * 1000)
        segment_size_s = segment_size_ms / 1000
        x = np.empty((0, 60))

        for offset in range(0, duration_ms, segment_size_ms):
            if offset + segment_size_ms <= duration_ms:
                y, sr = librosa.load(filename, sr=None, offset=offset / 1000, duration=segment_size_s)
                mfcc = librosa.feature.mfcc(y, sr)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)

                feature_vector = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta)).T

                x = np.concatenate((x, feature_vector))

        return x

    def _classify_frames_gen(self, filename):
        features = self._get_mfcc_with_deltas(filename, self.segment_size)

        for feature in features:
            yield self.gmm.predict(np.reshape(feature, (1, -1)))

    def classify_segments(self, segment_path):
        for root, _, filenames in os.walk(segment_path):
            for filename in sorted([f for f in filenames if f.endswith(ALLOWED_EXTENSIONS)],
                                   key=lambda name: int(os.path.splitext(name)[0])):

                n = 0
                speech_count = 0

                for is_speech in self._classify_frames_gen(os.path.join(root, filename)):
                    n += 1
                    if is_speech:
                        speech_count += 1

                yield speech_count > (n - speech_count)