import os
import pickle
from collections import defaultdict

import librosa
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common.utils.paths import get_speaker_pickle
from spdr.embedding_extractor import EmbeddingExtractor
from spdr.utils import SPDR_Util


def train_gmm(embedding_pickle_file):
    config = SPDR_Util.load_config()

    with open(os.path.join(config["data"]["pickle"], embedding_pickle_file), 'rb') as embedding_file:
        embeddings = pickle.load(embedding_file)

    cv_types = ['spherical', 'tied', 'diag', 'full']
    bic = []
    lowest_bic = np.infty

    for cv_type in cv_types:
        gmm = mixture.GaussianMixture(n_components=2, covariance_type=cv_type, max_iter=1000)
        gmm.fit(embeddings)

        bic.append(gmm.bic(embeddings))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    print("Lowest BIC is %f with covariance_type %s" % (lowest_bic, best_gmm.covariance_type))

    with open(os.path.join("data/vad_models", embedding_pickle_file), 'wb') as pFile:
        pickle.dump(best_gmm, pFile)

    for i, embedding in enumerate(embeddings):
        print("Embedding %d has label %d" % (i, best_gmm.predict(np.reshape(embedding, (1, -1)))))


def test_gmm_embeddings(embedding_pickle_file, model_pickle):
    config = SPDR_Util.load_config()

    with open(os.path.join("data/vad_models/", model_pickle), "rb") as model_file:
        gmm = pickle.load(model_file)

    with open(os.path.join(config["data"]["pickle"], embedding_pickle_file), 'rb') as embedding_file:
        embeddings = pickle.load(embedding_file)

    pca = PCA(75)
    reduced_emb = pca.fit_transform(embeddings)

    for i, embedding in enumerate(reduced_emb):
        print("Embedding %d has label %d" % (i, gmm.predict(np.reshape(embedding, (1, -1)))))


def train_gmm_by_speaker_pickle(speaker_pickle):
    config = SPDR_Util.load_config()

    if os.path.isfile("data/vad_models/voxceleb_emb.pkl"):
        with contextlib.closing(open("data/vad_models/voxceleb_emb.pkl", 'rb')) as embedding_file:
            embeddings = pickle.load(embedding_file)
    else:
        with contextlib.closing(open(get_speaker_pickle(speaker_pickle), 'rb')) as f:
            (X, y, speaker_names) = pickle.load(f)

        extractor = EmbeddingExtractor(
            n_segments=None,
            segment_path=None,
            network_file=config["embeddings"]["network"],
            max_audio_length=800,
            X=X)

        embeddings = extractor.extract_embeddings(None)
        with contextlib.closing(open("data/vad_models/voxceleb_emb.pkl", 'wb')) as embedding_file:
            pickle.dump(embeddings, embedding_file)

    pca = PCA(75)
    reduced_emb = pca.fit_transform(embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()
    plt.close(fig)

    cv_types = ['spherical', 'tied', 'diag', 'full']
    bic = []
    lowest_bic = np.infty

    for cv_type in cv_types:
        gmm = mixture.GaussianMixture(n_components=2, covariance_type=cv_type, max_iter=1000)
        gmm.fit(reduced_emb)

        bic.append(gmm.bic(reduced_emb))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    print("Lowest BIC is %f with covariance_type %s" % (lowest_bic, best_gmm.covariance_type))

    with open(os.path.join("data/vad_models", "vad_voxceleb.pkl"), 'wb') as pFile:
        pickle.dump(best_gmm, pFile)


def train_gmm_with_mfcc(training_data_path, valid_speaker_file):
    if os.path.isfile("data/vad_models/mfcc_features.pkl"):
        with contextlib.closing(open("data/vad_models/mfcc_features.pkl", 'rb')) as mfcc_file:
            X = pickle.load(mfcc_file)
    else:
        X = extract_mfcc(training_data_path, valid_speaker_file)
        with contextlib.closing(open("data/vad_models/mfcc_features.pkl", 'wb')) as mfcc_file:
            pickle.dump(X, mfcc_file)

    # pca = TSNE(n_components=2)
    # reduced = pca.fit_transform(X)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(reduced[:, 0], reduced[:, 1])
    # plt.show()
    # plt.close(fig)

    cv_types = ['spherical', 'tied', 'diag', 'full']
    bic = []
    lowest_bic = np.infty

    for cv_type in cv_types:
        gmm = mixture.GaussianMixture(n_components=2, covariance_type=cv_type, max_iter=1000)
        gmm.fit(X)

        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

    print("Lowest BIC is %f with covariance_type %s" % (lowest_bic, best_gmm.covariance_type))

    with open(os.path.join("data/vad_models", "vad_voxceleb.pkl"), 'wb') as pFile:
        pickle.dump(best_gmm, pFile)


def test_gmm_with_mfcc(segment_path, gmm_model):
    with open(os.path.join("data/vad_models/", gmm_model), "rb") as model_file:
        gmm = pickle.load(model_file)

    for root, _, filenames in os.walk(segment_path):
        for filename in sorted([f for f in filenames if f.endswith(".wav")],
                               key=lambda name: int(os.path.splitext(name)[0])):
            n = 0
            speech_count = 0

            for is_speech in classify_frames_gen(os.path.join(root, filename), 500, gmm):
                n += 1
                if is_speech:
                    speech_count += 1

            print("segment %s is %s speech" %
                  (os.path.splitext(filename)[0], "not" if speech_count < (n - speech_count) else ""))


def classify_frames_gen(filename, frame_duration_ms, gmm):
    features = get_mfcc_with_deltas(filename, frame_duration_ms)

    for feature in features:
        yield gmm.predict(np.reshape(feature, (1, -1)))


def extract_mfcc(training_data_path, valid_speaker_file):
    X = np.empty((0, 60))

    with contextlib.closing(open(valid_speaker_file)) as speaker_file:
        valid_speakers = speaker_file.readlines()

    valid_speakers = [speaker.replace("\n", "") for speaker in valid_speakers]

    gen_spectrogram_per_speaker = defaultdict(int)
    max_files_per_speaker = 45
    old_speaker = ""
    max_speakers = len(valid_speakers)
    curr_speaker_num = -1

    for root, directories, filenames in os.walk(training_data_path, followlinks=True):
        if valid_speakers and os.path.split(root)[1] not in valid_speakers:
            continue

        for filename in [filename for filename in filenames if filename.endswith(".wav")]:
            speaker = os.path.split(root)[1]
            if speaker != old_speaker:
                curr_speaker_num += 1
                old_speaker = speaker
                print('Extraction progress: %d/%d' % (curr_speaker_num + 1, max_speakers))

            if gen_spectrogram_per_speaker[speaker] < max_files_per_speaker:
                full_path = os.path.join(root, filename)
                X = np.concatenate((X, get_mfcc_with_deltas(full_path, segment_size_ms=500)))
                gen_spectrogram_per_speaker[speaker] += 1

    return X


def get_mfcc_with_deltas(filename, segment_size_ms):
    duration = librosa.get_duration(filename=filename)
    duration_ms = int(duration * 1000)
    segment_size_s = segment_size_ms / 1000
    x = np.empty((0, 60))

    for offset in range(0, duration_ms, segment_size_ms):
        if offset + segment_size_ms <= duration_ms:
            y, sr = librosa.load(filename, sr=None, offset=offset / 1000, duration=segment_size_s)
            mfcc = librosa.feature.mfcc(y, sr)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)

            feature_vector = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta)).T

            x = np.concatenate((x, feature_vector))

    return x


if __name__ == '__main__':
    test_gmm_with_mfcc("data/out/NIST_20080307-0955_d03_NONE", "vad_voxceleb.pkl")
    # train_gmm_with_mfcc("ZHAW_deep_voice/common/data/training/VoxCelebV1/TRAIN/",
    #                   "ZHAW_deep_voice/common/data/speaker_lists/speakers_voxceleb_speaker_diarization_100.txt")
    # test_gmm("EDI_20071128-1000_ci01_NONE_normalized.pkl", "vad_voxceleb.pkl")
    # train_gmm_by_speaker_pickle("speakers_voxceleb_speaker_diarization_100_cluster")
    # test_gmm("EDI_20071128-1000_ci01_NONE.pkl", "vad_voxceleb.pkl")
    # train_gmm("NIST_20080307-0955_d03_NONE.pkl")

