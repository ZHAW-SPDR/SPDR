import contextlib
import itertools
import pickle
import numpy as np

from spdr.clustering.dominantset import dominantset

PICKLE_SPEC = "data/pickles/TIMIT_PHRASES_10_SPEAKERS_01_spec.pkl"
PICKLE_EMBEDDINGS = "data/pickles/TIMIT_PHRASES_10_SPEAKERS_01_emb.pkl"
PICKLE_RESULTS = "data/pickles/GRIDSEARCH_DOMINANTSET.pkl"

if __name__ == '__main__':
    with contextlib.closing(open(PICKLE_SPEC, "rb")) as spec:
        spectrogram, speaker_matching = pickle.load(spec)

    with contextlib.closing(open(PICKLE_EMBEDDINGS, "rb")) as spec:
        X = pickle.load(spec)

    epsilons = [1 / 10**e for e in range(2, 12)]
    thetas = np.arange(0.0, 0.9995, 0.0005)
    mrs = {}
    lowest_mr = 1.0

    for (e, t) in itertools.product(epsilons, thetas):
        dos = dominantset.DominantSetClustering(feature_vectors=X, speaker_ids=np.asarray(speaker_matching),
                                                metric="cosine",
                                                dominant_search=False,
                                                epsilon=e, cutoff=(-1.0 * t), reassignment="noise")

        dos.apply_clustering()

        (mr, randi, acp) = dos.evaluate()
        mrs[(e, t)] = (mr, randi, acp)

        if mr < lowest_mr:
            print("NEW lowest mr found (=%f) with epsilon=%.11f and theta=%.11f" % (mr, e, t))
            lowest_mr = mr

        with contextlib.closing(open(PICKLE_RESULTS, "wb")) as r:
            pickle.dump(mrs, r)


