import numpy as np
import logging
from keras.models import Model
from keras.models import load_model
from ZHAW_deep_voice.common.utils.paths import *
from ZHAW_deep_voice.networks.pairwise_lstm.core.pairwise_kl_divergence import pairwise_kl_divergence
from ZHAW_deep_voice.networks.pairwise_lstm.core.data_gen import extract
import ZHAW_deep_voice.common.spectogram.spectrogram_converter as spectrogram_converter
from spdr.utils import get_filename_without_extension


class EmbeddingExtractor:

    ALLOWED_EXTENSIONS = ('wav', 'mp3')

    def __init__(self, segment_path, n_segments, max_audio_length, network_file, frequency_elements=128, X=None):
        self.segment_path = segment_path
        self.n_segments = n_segments
        self.max_audio_length = max_audio_length
        self.frequency_elements = frequency_elements
        self.logger = logging.getLogger(__name__)
        self.network_file = network_file
        self.generate_spectrogram = False

        if X is None:
            self.X = np.zeros((self.n_segments, 1, self.frequency_elements, self.max_audio_length), dtype=np.float32)
            self.generate_spectrogram = True
        else:
            self.X = X

    def extract_embeddings(self, files):
        if self.generate_spectrogram:
            self._extract_mel_spectrogram_per_segment(files)
        return self._run_pairwise_lstm_network()

    def _run_pairwise_lstm_network(self):
        best_checkpoint = list_all_files(get_experiment_nets(), self.network_file)[0]


        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = pairwise_kl_divergence
        custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
        optimizer = 'rmsprop'

        self.logger.info("Running checkpoint pairwise_lstm network")

        # Load and compile the trained network
        network_file = get_experiment_nets(best_checkpoint)
        model_full = load_model(network_file, custom_objects=custom_objects)
        model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        model_partial = Model(inputs=model_full.inputs, outputs=model_full.layers[2].output)

        X_test = self._prepare_input()
        network_input = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[2])
        embeddings = model_partial.predict(network_input)

        return embeddings

    def _prepare_input(self, segment_size=50):
        segments = self.X.shape[0] * (self.max_audio_length // segment_size)
        X_test = np.zeros((segments, 1, self.frequency_elements, segment_size), dtype=np.float32)

        pos = 0
        for i in range(len(self.X)):
            spect = extract(self.X[i, 0], segment_size)

            for j in range(int(spect.shape[1] / segment_size)):
                seg_idx = j * segment_size
                X_test[pos, 0] = spect[:, seg_idx:seg_idx + segment_size]
                pos += 1

        return X_test

    def _extract_mel_spectrogram_per_segment(self, audiofiles):
        """
        Extracts the mel-spectrogram per segment
        :return:
        X: the filled data in the 4D array [Segment, Channel, Frequency, Time]
        """
        
        global_idx = 0

        for audiofile in audiofiles:
            segment_folder = os.path.join(self.segment_path, get_filename_without_extension(os.path.basename(audiofile)))

            for root, _, filenames in os.walk(segment_folder):
                if global_idx >= self.n_segments:
                    break

                for filename in sorted([f for f in filenames if f.endswith(self.ALLOWED_EXTENSIONS)], 
                        key=lambda name: int(os.path.splitext(name)[0])):
                    if global_idx >= self.n_segments:
                        break

                    full_path = os.path.join(root, filename)
                    self.logger.info("Extract mel_spectrogram for segment %d (%s)" % (global_idx, filename))

                    Sxx = spectrogram_converter.mel_spectrogram(full_path)

                    for i in range(Sxx.shape[0]):
                        for j in range(Sxx.shape[1]):
                            self.X[global_idx, 0, i, j] = Sxx[i, j]

                    self.logger.info("Extracted mel_spectrogram for segment %d successfully" % global_idx)

                    global_idx += 1

            if global_idx < self.n_segments:
                self.X = np.resize(self.X, (global_idx, self.X.shape[1], self.X.shape[2], self.X.shape[3]))
                
            self.logger.info("Extract mel_spectrogram for %d segments" % global_idx)
            global_idx = 0
