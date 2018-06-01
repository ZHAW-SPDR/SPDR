import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from spdr.clustering.dominantset import dominantset

def clustering():
    c = 2
    n = 100

    f, axes = plt.subplots(3, 1)
    np.random.seed(1234891023)

    c1 = np.random.normal(5.0, 1.0, (n, c))
    c2 = np.random.normal(20.0, 2.0, (n, c))

    features = np.concatenate((c1, c2))
    cutoffs = [1.0e-3] 
    epsilons = [1.0e-1]
    reassignments = ['single']
    metrics = ['euclidean']
    dom_search = [False]
    best = c*n
    best_config = None
    counter = 0
    while counter is not 30:
        for eps in epsilons:
            for co in cutoffs:
                for r in reassignments:
                    for m in metrics:
                        for ds in dom_search:
                            counter += 1
                            dos = dominantset.DominantSetClustering(feature_vectors=features,
                                                                    speaker_ids=np.array(([0] * n) + ([1] * n)),
                                                                    metric=m, dominant_search=ds,
                                                                    reassignment=r,
                                                                    epsilon=eps, cutoff=co)

                            dos.apply_clustering()
                            clusters = dos.get_n_clusters()
                            summary = "Epsilon: %s \nCutoff: %s \nReassignment: %s\nMetric: %s\nDominant Search: %s\n => Clusters: %s" % (str(eps), str(co), r, m, str(ds), str(clusters))
                            if clusters == c:
                                print("Best found!")
                                break
                            if best >= clusters and clusters != 0 and clusters != 1:
                                best_config = summary
                                best = clusters
                            if clusters < 30:
                                print(summary)
                                print(dos.ds_result)
                                print("-"*40)
    print("\nBest config:\n%s" % (best_config))
    axes[0].scatter(features[:, 0], features[:, 1], c=dos.ds_result)

    Z = linkage(features, 'complete')
    dendrogram(Z)

    clusters = fcluster(Z, 15.0, "distance")
    axes[1].scatter(features[:, 0], features[:, 1], c=clusters)

    plt.show()


if __name__ == '__main__':
    clustering()
