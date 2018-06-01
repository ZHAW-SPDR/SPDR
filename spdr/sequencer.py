"""
Sequencer for speaker diarization.

This sequencer joins pieces of wav files and detectes speaker changes.
Embedding dimensionality reduction is being performed using tSNE.


Usage: Use in code
    sequencer = SPDR_Sequencer(embeddings)
    sequencer.run(plot_only=True)

Sequencer for Speaker diarization
optional arguments:
  -h, --help  show this help message and exit
  
"""
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise as pw
from scipy.stats.stats import pearsonr, spearmanr, kendalltau
from pydub import AudioSegment
from .utils import SPDR_Util, Sequence
from scipy.spatial.distance import cosine, euclidean, pdist, squareform
from sklearn import preprocessing
from .cluster import SPDR_Clustering


class SPDR_Sequencer():

    def __init__(self, embeddings, config=None):
        self.config = config if config else SPDR_Util.load_config()
        self.embeddings = embeddings
        self.reduced_embeddings = None
        self.use_PCA = self.config['sequence']['use_PCA']
        self.metric = self.config['sequence']['metric']
        self.segment_path = self.config['data']['out']
        self.segment_duration_ms = self.config['segment']['size']
        self.cutoff = float(self.config['sequence']['cutoff'])
        self.linkage = self.config['sequence']['linkage']
        # Fixing random state for reproducibility
        self.random_state = np.random.seed(19680801)

    def perform_dim_reduction_for(self, embeddings, n_components=2):
        if self.use_PCA:
            pca = PCA(n_components)
            results = pca.fit_transform(embeddings)
        else:
            tsne = TSNE(n_components, verbose=1, perplexity=40, n_iter=2500, init='pca')
            results = tsne.fit_transform(embeddings)

        return results

    def perform_dim_reduction(self, n_components=2):
        if self.reduced_embeddings is not None:
            return

        self.reduced_embeddings = self.perform_dim_reduction_for(self.embeddings, n_components)

    def plot(self, figure_name=None):
        method = 'PCA' if self.use_PCA else 'tSNE'
        N = len(self.reduced_embeddings)

        X = [i[0] for i in self.reduced_embeddings]
        Y = [i[1] for i in self.reduced_embeddings]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X, Y, c=list(range(0, N)))

        for i, xyz in enumerate(zip(X, Y)):
            ax.text(xyz[0], xyz[1], '%s' % (str(i)), size=10, zorder=1)

        plt.title('Dimension reduction using ' + method)

        if not figure_name:
            plt.savefig('figure_' + method + '.png')
        else:
            plt.savefig(figure_name)

        return None

    @staticmethod
    def plot_sequences(sequence_embeddings, cluster_result):
        if sequence_embeddings.shape[1] > 2:
            return

        plt.figure()
        plt.scatter(sequence_embeddings[:, 0], sequence_embeddings[:, 1], c=cluster_result)
        plt.title('Sequences')
        plt.savefig('sequences.png')

    def _distance(self, u, v):
        return pdist([u, v], metric=self.metric.lower())

    def _get_max_distance_in_vector(self, X):
        return max(pdist(X, metric=self.metric.lower()))

    def find_changepoints(self):
        similarity = [0]
        seen_embeddings = [self.embeddings[0]]
        for embedding in self.embeddings[1:]:
            distances = []
            for seen_embedding in seen_embeddings:
                distance = self._distance(seen_embedding, embedding)[0]
                distances.append(distance)
            
            best_dist = float("inf")
            if self.linkage == 'single':
                best_dist = min(distances)
            elif self.linkage == 'complete':
                best_dist = max(distances)
            elif self.linkage == 'last':
                best_dist = distances[-1]
            else:
                raise Exception("Not supported linkage")

            vec_elem = 0 if best_dist > self.cutoff else 1
            if vec_elem == 0:
                seen_embeddings = [embedding]
            else:
                seen_embeddings.append(embedding)
            similarity.append(vec_elem)

        # change point detector
        previous = None
        changepoints = []

        for i, e in enumerate(similarity):
            if previous == 1 and e == 0:
                print("Speaker Change detected at %d" % (i * 500))
                changepoints.append(0)
            else:
                changepoints.append(1)

            previous = e

        # changepoints.append(changepoints[-1])

        return changepoints
