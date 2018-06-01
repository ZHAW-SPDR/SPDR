import contextlib
import os
import pickle
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from spdr.clustering.dominantset import dominantset
from spdr.embedding_extractor import EmbeddingExtractor
from spdr.utils import SPDR_Util


class DominantSetExperiment:
    def __init__(self, experiment_name):
        self.config = SPDR_Util.load_config()
        self.experiment_name = experiment_name

    def __call__(self, *args, **kwargs):
        print("Run experiment %s..." % self.experiment_name)

        for i, spectrogram_pickle in enumerate([os.path.join(self.config["data"]["pickle"], experiment_file)
                                                for experiment_file in os.listdir(self.config["data"]["pickle"])
                                                if experiment_file.startswith(self.experiment_name)
                                                   and experiment_file.endswith("_spec.pkl")]):

            with contextlib.closing(open(spectrogram_pickle, "rb")) as spec:
                spectrogram, speaker_matching = pickle.load(spec)

            emb_pickle_file = spectrogram_pickle.replace("_spec.pkl", "_emb.pkl")

            if os.path.isfile(emb_pickle_file):
                with contextlib.closing(open(emb_pickle_file, "rb")) as spec:
                    X = pickle.load(spec)
            else:
                embedding_extractor = EmbeddingExtractor(
                    segment_path=None, n_segments=None,
                    network_file="pairwise_lstm_100_best.h5",
                    max_audio_length=int(self.config['segment']['size'] / 10) + 1,
                    X=spectrogram
                )

                X = embedding_extractor.extract_embeddings(None)
                with contextlib.closing(open(emb_pickle_file, "wb")) as spec:
                    pickle.dump(X, spec)

            unique_speaker_names = list(np.unique(speaker_matching))
            speaker_map = np.array([unique_speaker_names.index(speaker) for speaker in speaker_matching])

            dos = dominantset.DominantSetClustering(feature_vectors=X, speaker_ids=np.array(speaker_matching),
                                                    metric=self.config["clustering"]["metric"],
                                                    dominant_search=False,
                                                    epsilon=1e-6, cutoff=-0.15, reassignment="noise")

            dos.apply_clustering()
            unique_cluster_result = np.unique(dos.ds_result)

            if X.shape[0] <= 30:
                reducer = TSNE(2, verbose=1, perplexity=5.0, n_iter=2500, init='random')
            else:
                reducer = TSNE(2, verbose=1, perplexity=31.0, n_iter=2500, init='random')

            reduced_embeddings = reducer.fit_transform(X)

            fig = plt.figure(figsize=(8, 8))
            axs = fig.subplots(nrows=2, ncols=2)
            fig.suptitle("Clustering: %s_0%d" % (self.experiment_name, i + 1))

            axs[0, 0].set_title("Ground truth")
            for idx, speaker in enumerate(unique_speaker_names):
                indexes = np.where(speaker_map == idx)[0]
                axs[0, 0].scatter(reduced_embeddings[indexes, 0], reduced_embeddings[indexes, 1],
                                  label=speaker)

            axs[0, 1].set_title("Clustered")
            for speaker in unique_cluster_result:
                indexes = np.where(dos.ds_result == speaker)[0]
                axs[0, 1].scatter(reduced_embeddings[indexes, 0], reduced_embeddings[indexes, 1],
                                  label=speaker)

            axs[1, 0].set_frame_on(False)
            axs[1, 0].get_xaxis().set_visible(False)
            axs[1, 0].get_yaxis().set_visible(False)

            handles, labels = axs[0, 0].get_legend_handles_labels()
            axs[1, 0].legend(handles, labels, loc="upper center", fancybox=True, shadow=True, ncol=2)

            axs[1, 1].set_frame_on(False)
            axs[1, 1].get_xaxis().set_visible(False)
            axs[1, 1].get_yaxis().set_visible(False)

            handles, labels = axs[0, 1].get_legend_handles_labels()
            axs[1, 1].legend(handles, labels, loc="upper center", fancybox=True, shadow=True, ncol=2)

            plt.savefig("./data/plots/dominantset/Clustering_%s_0%d.png" % (self.experiment_name, i + 1))
            plt.close(fig)


if __name__ == '__main__':
    def experiments():
        yield DominantSetExperiment("TIMIT_SINGLE_SPEAKER")
        yield DominantSetExperiment("TIMIT_ONE_PHRASE_5_SPEAKERS")
        yield DominantSetExperiment("TIMIT_PHRASES_10_SPEAKERS")
        yield DominantSetExperiment("VOXCELEB_SINGLE_SPEAKER")
        yield DominantSetExperiment("VOXCELEB_ONE_PHRASE_5_SPEAKERS")
        yield DominantSetExperiment("VOXCELEB_PHRASES_10_SPEAKERS")
        yield DominantSetExperiment("RT09_SINGLE_SPEAKER")
        yield DominantSetExperiment("RT09_ONE_PHRASE_5_SPEAKER")
        yield DominantSetExperiment("RT09_PHRASES_10_SPEAKERS")


    for experiment in experiments():
        experiment()
