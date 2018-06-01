import numpy as np

from spdr.clustering.dominantset import dominantset
from sklearn.metrics import pairwise as pw
from scipy.spatial import distance
from scipy.special import comb
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, ClusterNode
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib import cm


class ClusteringController:

    def __init__(self, config):
        self.config = config
        self.algorithm = self.config["clustering"]["algorithm"]
        self.metric = self.config["clustering"]["metric"]

        self.clusteringAlgorithm = {
            "dominantset": self._run_dominant_set,
            "hierarchical": self._run_hierarchical
        }

    @staticmethod
    def _convert_to_similarity_vector(changepoint_vector):
        n = len(changepoint_vector)
        n_comb = comb(n, 2, exact=True)
        similarity_vector = np.ones(n_comb, dtype=np.int32)

        indexes = range(len(changepoint_vector))
        for idx, (i, j) in enumerate(zip(indexes, indexes[1:])):
            # according the scipy documentation, the index of the pair (i, j) can be calculated according this formula:
            # v[{n choose 2}-{n - i choose 2} + (j - i - 1)]
            similarity_vector[n_comb - comb(n - i, 2, exact=True) + (j - i - 1)] = changepoint_vector[idx]

        return similarity_vector

    @staticmethod
    def _convert_to_similarity_lists(changepoint_vector):
        similarity_lists = []
        similarity_list = []

        for idx, is_changepoint in enumerate(changepoint_vector):
            if is_changepoint == 0 or idx == len(changepoint_vector) - 1:
                if len(similarity_list) > 0:
                    similarity_lists.append(similarity_list)
                    similarity_list = []

                similarity_lists.append(idx)
            else:
                similarity_list.append(idx)

        return similarity_lists

    def plot_similarity_matrix(self, embeddings):
        if self.metric == "cosine":
            dist_mat = pw.cosine_distances(embeddings)
            # dist_mat = pw.cosine_similarity(embeddings)
            # dist_mat = np.arccos(dist_mat)
            # dist_mat[np.eye(dist_mat.shape[0]) > 0] = 0
            # dist_mat /= np.pi
        else:
            dist_mat = distance.pdist(embeddings, metric=self.metric)
            dist_mat = distance.squareform(dist_mat)

        sigmas = np.sort(dist_mat, axis=1)[:, 1:8]
        sigmas = np.mean(sigmas, axis=1)
        sigmas = np.dot(sigmas[:, np.newaxis], sigmas[np.newaxis, :])
        dist_mat /= -sigmas
        dist_mat = np.exp(dist_mat)

        dist_mat = dist_mat * (1. - np.identity(dist_mat.shape[0]))

        cmap = cm.get_cmap('inferno')
        cax = plt.matshow(dist_mat, interpolation='nearest', cmap=cmap)
        plt.grid(True)
        plt.title('Sequence similarity matrix using metric %s' % self.metric)
        plt.colorbar(cax)
        plt.show()

    def cluster_embeddings(self, embeddings, speaker_ids, changepoint_vector):
        algorithm = self.algorithm if self.algorithm in self.clusteringAlgorithm else "dominantset"

        return self.clusteringAlgorithm[algorithm](embeddings, speaker_ids, changepoint_vector)

    def _run_dominant_set(self, embeddings, speaker_ids, changepoint_vector):
        similarity_list = None
        if changepoint_vector is not None:
            similarity_list = ClusteringController._convert_to_similarity_lists(changepoint_vector)

        epsilon = 1e-6 if not self.config["clustering"]["dominantset"] \
            else float(self.config["clustering"]["dominantset"]["epsilon"])

        cutoff = 0.2 if not self.config["clustering"]["dominantset"] \
            else float(self.config["clustering"]["dominantset"]["cutoff"])

        apply_similarity_mode = "full" if not self.config["clustering"]["dominantset"] \
            else self.config["clustering"]["dominantset"]["apply_similarity_mode"]

        dos = dominantset.DominantSetClustering(feature_vectors=embeddings, speaker_ids=speaker_ids,
                                                metric=self.metric, dominant_search=False,
                                                epsilon=epsilon, cutoff=-1.0 * cutoff, reassignment="single",
                                                similarity_list=similarity_list,
                                                apply_similarity_mode=apply_similarity_mode)

        dos.apply_clustering()

        return dos.ds_result

    def _run_hierarchical(self, embeddings, speaker_ids, changepoint_vector):
        embeddings_distance = cdist(embeddings, embeddings, self.metric)
        embeddings_linkage = linkage(embeddings_distance, "ward", self.metric)

        plt.figure()
        dendrogram(embeddings_linkage)
        plt.show()

        threshold = 350.0
        return fcluster(embeddings_linkage, threshold, 'distance')
