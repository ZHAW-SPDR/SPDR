import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .utils import SPDR_Util, Sequence
from sklearn import preprocessing, metrics
from .affinity_clustering import SPDR_Affinity
from .hdbscan_clustering import SPDR_HDBSCAN
import random
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine, euclidean, pdist, squareform

class SPDR_Clustering():

    class Cluster_Result(object):
        def __init__(self, labels, clusters, silhouette_score, configuration):
            self.labels = labels
            self.clusters = clusters
            self.silhouette_score = silhouette_score
            self.configuration = configuration

    def __init__(self):
        self.config = SPDR_Util.load_config()
        self.metric = self.config['sequence']['metric']
        self.do_finish = self.config['clustering']['affinity_intervals_finish_all']
        self.use_ensemble = self.config['clustering']['affinity_use_ensemble']
        self.loaded = False
        self.converted = False
    
    def load(self, embeddings):
        self.embeddings = embeddings
        self._do_convert()
        self.loaded = True

    def _do_convert(self):
        self.X_simmat = self._convert_to_similarity_matrix(self.embeddings)
        self.X_distmat = self._convert_to_distance_matrix(self.embeddings)
        self.converted = True

    def _convert_to_similarity_matrix(self, embeddings, metric=None):
        # if self.metric.lower() == 'euclidean':
        if metric is None:
            metric = self.metric.lower()
        #return squareform(pdist(embeddings,metric=metric))
        return pairwise_distances(embeddings, metric=metric)

        # else:  # cosine distance
        #     sim_mat = pdist(embeddings, metric=self.metric)
        #     sim_mat = squareform(sim_mat)
        #     normalized_simmat = preprocessing.normalize(sim_mat)
        #     sim_mat = np.arccos(normalized_simmat)
        #     sim_mat[np.eye(sim_mat.shape[0]) > 0] = 0
        #     sim_mat /= np.pi
        #     return sim_mat

    def _convert_to_distance_matrix(self, embeddings):
        simmat = self._convert_to_similarity_matrix(embeddings)
        normalized_simmat = preprocessing.normalize(simmat)
        return 1 - normalized_simmat

    def _cluster(self):
        
        ensemble_params = {
            'damping': np.arange(0.5, 1.0, 0.1),
            'data': [
                {'X':self._convert_to_similarity_matrix(embeddings=self.embeddings, metric='euclidean'), 'M':'euclidean'},
                {'X':self._convert_to_similarity_matrix(embeddings=self.embeddings, metric='cosine'), 'M':'cosine'}
            ]
        }
        regular_params = {
            'damping': [0.5],
            'data': [
                {'X':self._convert_to_similarity_matrix(embeddings=self.embeddings, metric=self.metric.lower()), 'M':self.metric.lower()}
            ]
        }
        params = ensemble_params if self.use_ensemble else regular_params
        configs = []
        for d in params['damping']:
            for p in params['data']:
                configs.append((d,p))
        votable_results = []
        for i in range(0, len(configs)):
            print('\tConfig %d of %d:' % (i+1, len(configs)))
            damping_factor = round(configs[i][0],1)
            data = configs[i][1]['X']
            algorithm = 'affinity'
            if algorithm.lower() == 'hdbscan':
                data = data.astype(np.float64)
                labels, n_clusters_ = SPDR_HDBSCAN.cluster(X=data)
            elif algorithm.lower() == 'affinity':
                labels, n_clusters_ = SPDR_Affinity.cluster(X=data, damping=damping_factor)

            
            labels = [label if label >= 0 else 1000 for label in labels]
            
            #n_clusters_ = len(cluster_centers_indices)
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters_ == 1:
                print('Skipping config, no clusters detected!')
                continue

            print('\t - Configuration:')
            print('\t\t - Damping Factor: %.1f' % damping_factor)
            print('\t\t - Metric: %s' % configs[i][1]['M'])
            print('\t - Estimated number of clusters: %d' % n_clusters_)
            sil_score = metrics.silhouette_score(self.X_simmat, labels, metric='precomputed')
            print('\t - Silhouette Coefficient: %0.3f' % sil_score)
            votable_results.append(self.Cluster_Result(
                labels=labels,
                clusters=n_clusters_,
                silhouette_score=sil_score,
                configuration=configs[i]
            ))
        data = [r.labels for r in votable_results]
        voted_labels = []
        for i in range(0,np.array(data).shape[1]):
            elems = []
            for j in range(0,np.array(data).shape[0]):
                elems.append(data[j][i])
            l = dict((x,elems.count(x)) for x in set(elems))
            voted_labels.append(max(l, key=l.get)) # missing should be randomized
        
        voted_labels = np.array(voted_labels)
        recalculated_sil_score = metrics.silhouette_score(self.X_simmat, voted_labels, metric='precomputed')
        print(' - Combined Silhouette Coefficient: %0.3f' % recalculated_sil_score)
        return voted_labels, recalculated_sil_score

    def do_cluster(self, intervals=1):
        prev_best_sil_score = -1
        for i in range(0,intervals):
            print('Affinity Interval %d of %d:' % (i+1, intervals))
            if not self.converted:
                self._do_convert()
            
            labels, sil_score = self._cluster()
            if sil_score > prev_best_sil_score or self.do_finish:
                current_label = -1
                current_sequence = []
                clusters = []
                for idx in range(0, len(labels)+1):
                    if idx == len(labels):
                        # cleanup
                        clusters.append(current_sequence)
                        mean = np.mean(self.embeddings[current_sequence], axis=0)
                        for i in range(0, len(current_sequence)):
                            self.embeddings[current_sequence] = mean
                        continue
                    label = labels[idx]
                    if idx == 0:
                        current_label = label
                        current_sequence.append(idx)
                        continue            
                    if label != current_label:
                        # start new sequence
                        clusters.append(current_sequence)
                        mean = np.mean(self.embeddings[current_sequence], axis=0)
                        for i in range(0, len(current_sequence)):
                            self.embeddings[current_sequence] = mean
                        current_sequence = []
                        current_sequence.append(idx)
                        current_label = label
                    else:
                        current_sequence.append(idx)
                self.clusters = clusters
                self.labels = labels
                self.silhouette_score = sil_score
                self.converted = False
                prev_best_sil_score = sil_score
            else:
                print('No improvement... Skipping!')
                break
        
        cluster_result = []
        for cluster in self.clusters:
            l = self.labels[cluster]
            cluster_result.append(np.argmax(np.bincount(l)))

        self.cluster_result = np.array(cluster_result)


    
    def _get_labels(self):
        return self.labels
    
    def _get_silhouette_score(self):
        return self.silhouette_score

    def _get_clusters(self):
        return self.clusters
    
    def get_results(self):
        return self.labels, self.silhouette_score, self.cluster_result, self.clusters

    def get_embeddings(self):
        return self.embeddings
