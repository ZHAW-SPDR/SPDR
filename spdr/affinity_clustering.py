from sklearn.cluster import AffinityPropagation

class SPDR_Affinity():

    @staticmethod    
    def cluster(X, damping):
        af = AffinityPropagation(affinity='precomputed', damping=damping)
        af.fit(X)
        labels = af.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, n_clusters_

