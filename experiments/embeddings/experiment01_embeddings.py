import os
import pickle
import contextlib

import numpy as np
from enum import Enum

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from spdr.embedding_extractor import EmbeddingExtractor
from spdr.utils import SPDR_Util


class Network(Enum):
    def __str__(self):
        return str(self.name)

    TIMIT_BLSTM = 1
    VOXCELEB_BLSTM = 2
    RT09_BLSTM = 3


class EmbeddingExperiment:
    network_checkpoints = {
        Network.TIMIT_BLSTM: "pairwise_lstm_100_best.h5",
        Network.RT09_BLSTM: "pairwise_lstm_transfer_best.h5",
        Network.VOXCELEB_BLSTM: "pairwise_lstm_1251_voxceleb_best.h5"
    }

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.config = SPDR_Util.load_config()

    def run_experiment(self):
        print("Run experiment %s..." % self.experiment_name)

        for i, spectrogram_pickle in enumerate([os.path.join(self.config["data"]["pickle"], experiment_file)
                                                for experiment_file in os.listdir(self.config["data"]["pickle"])
                                                if experiment_file.startswith(self.experiment_name)
                                                and experiment_file.endswith("_spec.pkl")]):

            print("Create plots for file %s..." % spectrogram_pickle)

            with contextlib.closing(open(spectrogram_pickle, 'rb')) as pFile:
                X, speaker_matching = pickle.load(pFile)

                unique_speaker_names = list(np.unique(speaker_matching))
                speaker_map = np.array([unique_speaker_names.index(speaker) for speaker in speaker_matching])

                fig = plt.figure(figsize=(12, 8))
                plt.suptitle("Embeddings: %s_0%d" % (self.experiment_name, i + 1))
                last_ax = None

                for j, (lstm_network_type, checkpoint) in enumerate(sorted(self.network_checkpoints.items(),
                                                                           key=lambda item: item[0].value)):
                    print("Using %s..." % lstm_network_type)

                    embedding_extractor = EmbeddingExtractor(
                        segment_path=None, n_segments=None,
                        network_file=checkpoint,
                        max_audio_length=int(self.config['segment']['size'] / 10) + 1,
                        X=X
                    )

                    embeddings = embedding_extractor.extract_embeddings(None)

                    if embeddings.shape[0] <= 30:
                        reducer = TSNE(2, verbose=1, perplexity=5.0, n_iter=2500, init='random')
                    else:
                        reducer = TSNE(2, verbose=1, perplexity=31.0, n_iter=2500, init='random')

                    reduced_embeddings = reducer.fit_transform(embeddings)

                    last_ax = ax = fig.add_subplot(230 + (j + 1))

                    for idx, speaker in enumerate(unique_speaker_names):
                        indexes = np.where(speaker_map == idx)[0]
                        ax.scatter(reduced_embeddings[indexes, 0], reduced_embeddings[indexes, 1],
                                   label=speaker)

                    plt.title('%s' % lstm_network_type)
                    print("Done")

                handles, labels = last_ax.get_legend_handles_labels()

                ax = fig.add_subplot(235)
                ax.set_frame_on(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ncol = 5

                if self.experiment_name.startswith("TIMIT_"):
                    ncol = 6
                elif self.experiment_name.startswith("VOXCELEB_"):
                    ncol = 4

                ax.legend(handles, labels, loc="upper center", fancybox=True, shadow=True, ncol=ncol)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                    wspace=0.3, hspace=None)

                plt.savefig("./data/plots/embeddings/%s_0%d.png" % (self.experiment_name, i + 1))
                plt.close(fig)

                print("Created plots for file %s" % spectrogram_pickle)

        print("Run experiment %s successfully" % self.experiment_name)


if __name__ == '__main__':
    experiments = [
        EmbeddingExperiment("TIMIT_SINGLE_SPEAKER"),
        EmbeddingExperiment("TIMIT_ONE_PHRASE_5_SPEAKERS"),
        EmbeddingExperiment("TIMIT_PHRASES_10_SPEAKERS"),
        EmbeddingExperiment("VOXCELEB_SINGLE_SPEAKER"),
        EmbeddingExperiment("VOXCELEB_ONE_PHRASE_5_SPEAKERS"),
        EmbeddingExperiment("VOXCELEB_PHRASES_10_SPEAKERS"),
        EmbeddingExperiment("RT09_SINGLE_SPEAKER"),
        EmbeddingExperiment("RT09_ONE_PHRASE_5_SPEAKER"),
        EmbeddingExperiment("RT09_PHRASES_10_SPEAKERS"),
        EmbeddingExperiment("RT09_NORMALIZED_PHRASES_10_SPEAKERS")
    ]

    for experiment in experiments:
        experiment.run_experiment()
