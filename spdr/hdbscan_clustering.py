from hdbscan import HDBSCAN

class SPDR_HDBSCAN():
    
    @staticmethod
    def cluster(X):
        hdbs = HDBSCAN(metric='precomputed')
        hdbs.fit(X)
        labels = hdbs.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, n_clusters_

