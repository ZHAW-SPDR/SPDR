import random
import numpy as np
from pydub import AudioSegment
import pickle
import librosa
from keras.models import Model
from keras.models import load_model
from ZHAW_deep_voice.common.utils.paths import *
from ZHAW_deep_voice.networks.pairwise_lstm.core.data_gen import extract
from ZHAW_deep_voice.networks.pairwise_lstm.core.pairwise_kl_divergence import pairwise_kl_divergence

IDENTIFIER = 'TIMIT_PHRASES_10_SPEAKERS_05_'
PICKLEFOLDER = 'data/pickles/'

SAMPLES_PER_SPEAKER = 8
SPEAKER_SAMPLES = 10
SUB_SAMPLES_PER_SAMPLE = 4
SHUFFLE_SUB_SAMPLES = False

np.random.seed(42)

# Functions

def dyn_range_compression(x):
    return np.log10(1 + 10000 * x)

def mel_spectrogram(wav_file):
    # Read out audio range and sample rate of wav file
    audio_range, sample_rate = librosa.load(path=wav_file, sr=None)
    nperseg = int(10 * sample_rate / 1000)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_range, sr=sample_rate, n_fft=1024, hop_length=nperseg)

    # Compress the mel spectrogram to the human dynamic range
    for i in range(mel_spectrogram.shape[0]):
        for j in range(mel_spectrogram.shape[1]):
            mel_spectrogram[i, j] = dyn_range_compression(mel_spectrogram[i, j])
    return mel_spectrogram

def _prepare_input(X, max_audio_length, segment_size=50):
    segments = X.shape[0] * (max_audio_length // segment_size)
    X_test = np.zeros((segments, 1, 128, segment_size), dtype=np.float32)

    pos = 0
    for i in range(len(X)):
        spect = extract(X[i, 0], segment_size)

        for j in range(int(spect.shape[1] / segment_size)):
            seg_idx = j * segment_size
            X_test[pos, 0] = spect[:, seg_idx:seg_idx + segment_size]
            pos += 1

    return X_test

# DATASET
timit = 'ZHAW_deep_voice/common/data/training/TIMIT/TEST/'
voxceleb = 'ZHAW_deep_voice/common/data/training/VoxCelebV1/TRAIN/'
rt09 = 'ZHAW_deep_voice/common/data/training/RT09/TRAIN/'
rt09_EDI = 'ZHAW_deep_voice/common/data/training/RT09/TRAIN/EDI'

dataset = timit

# pick 500 random samples of 500ms length
possible_samples = []
for root, _, filenames in os.walk(dataset, followlinks=True):
    files = [x for x in filenames if x.endswith('RIFF.WAV')
             and len(AudioSegment.from_file(os.path.join(root, x), format='wav')) >= SUB_SAMPLES_PER_SAMPLE * 500]

    if len(files) >= SAMPLES_PER_SPEAKER:
        sample_idx = random.sample(range(0, len(files)), SAMPLES_PER_SPEAKER)
        subset = np.array(files)[sample_idx]
        possible_samples.append({'root': root, 'speaker': root.split('/')[-1], 'samples': subset})

speakers_idx = random.sample(range(0, len(possible_samples)), SPEAKER_SAMPLES)
print("using speakers: %a" % speakers_idx)

print('Spectrogram extraction')
spectrogram_pickle = os.path.join(PICKLEFOLDER, IDENTIFIER+'spec.pkl')
if os.path.isfile(spectrogram_pickle):
    with open(spectrogram_pickle, 'rb') as pFile:
        X, speaker_matching = pickle.load(pFile)
else:
    global_idx = 0
    _n_segments = 50000
    X = np.zeros((_n_segments, 1, 128, int(500 / 10) + 1), dtype=np.float32)
    speaker_matching = []
    # Spektrogram extraction
    for elem in np.array(possible_samples)[speakers_idx]:
        speaker = elem['speaker']
        for sample in elem['samples']:
            sample_file = os.path.join(elem['root'], sample)
            audio = AudioSegment.from_file(sample_file, format='wav')

            for i in range(0, SUB_SAMPLES_PER_SAMPLE):
                if SHUFFLE_SUB_SAMPLES:
                    audio_slice = audio[500:-500]
                    if len(audio_slice) <= 500:
                        continue

                    window_found = False
                    start = 0
                    while not window_found:
                        start_candidate = np.random.randint(low=0, high=len(audio_slice))
                        if start_candidate + 500 < len(audio_slice):
                            window_found = True
                            start = start_candidate
                    sub_slice = audio_slice[start:start + 500]
                else:
                    start = i * 500
                    sub_slice = audio[start:start + 500]

                sub_slice.export('tmp.wav', format='wav')

                # get spectrogram
                Sxx = mel_spectrogram('tmp.wav')
                os.remove('tmp.wav')
                for i in range(Sxx.shape[0]):
                    for j in range(Sxx.shape[1]):
                        X[global_idx, 0, i, j] = Sxx[i, j]
                speaker_matching.append(speaker)
                global_idx += 1
    if global_idx < _n_segments:
        X = np.resize(X, (global_idx, X.shape[1], X.shape[2], X.shape[3]))
    with open(spectrogram_pickle, 'wb') as pFile:
        pickle.dump((X, speaker_matching), pFile)

print('Embedding extraction')
# Embedding extraction
embedding_pickle = os.path.join(PICKLEFOLDER, IDENTIFIER+'emb.pkl')
if os.path.isfile(embedding_pickle):
    with open(embedding_pickle, 'rb') as pFile:
        embeddings = pickle.load(pFile)
else:
    best_checkpoint = list_all_files(get_experiment_nets(), "pairwise_lstm_1251_voxceleb_best.h5")[0]

    metrics = ['accuracy', 'categorical_accuracy', ]
    loss = pairwise_kl_divergence
    custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
    optimizer = 'rmsprop'

    # Load and compile the trained network
    network_file = get_experiment_nets(best_checkpoint)
    model_full = load_model(network_file, custom_objects=custom_objects)
    model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model_partial = Model(inputs=model_full.inputs, outputs=model_full.layers[2].output)

    X_test = _prepare_input(X, int(500 / 10) + 1)
    network_input = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[2])
    embeddings = model_partial.predict(network_input)
    with open(embedding_pickle, 'wb') as pFile:
        pickle.dump(embeddings, pFile)

print('Dataset export')
dataset_pickle = os.path.join(PICKLEFOLDER, IDENTIFIER+'dataset.pkl')
if os.path.isfile(dataset_pickle):
    print('Dataset already exported.')
else:
    dataset = []
    for elem in zip(embeddings, speaker_matching):
        dataset.append(elem)
    with open(dataset_pickle, 'wb') as pFile:
        pickle.dump(dataset, pFile)
    print('Export completed')