if __name__ == '__main__':
    from .dominantset import DominantSetClustering
    import numpy as np

    # have to be defined
    feature_vectors = np.zeros((5, 512))
    cluster_ids = []

    dos = DominantSetClustering(feature_vectors=feature_vectors, speaker_ids=cluster_ids,
                                   metric='cosine', dominant_search=False,
                                   epsilon=1e-6, cutoff=-0.1)
    dos.apply_clustering()
    print(dos.evaluate())  # MR - ARI - ACP
