import os
import random
import numpy as np
from shutil import copyfile

IN = 'TRAIN/'
OUT = 'SUBSET/'
SAMPLES_PER_SPEAKER = 3
SPEAKER_SAMPLES = 10

possible_samples = []
for root, _, filenames in os.walk(IN):
    files = [x for x in filenames if x.endswith('RIFF.WAV')]
    if len(files) > 0:
        sample_idx = random.sample(range(0, len(files)), SAMPLES_PER_SPEAKER)
        subset = np.array(files)[sample_idx]
        possible_samples.append({'root': root, 'speaker': root.split('/')[-1], 'samples': subset})

speakers_idx = random.sample(range(0, len(possible_samples)), SPEAKER_SAMPLES)
for elem in np.array(possible_samples)[speakers_idx]:
    for sample in elem['samples']:
        FROM = os.path.join(elem['root'], sample)
        DESTNAME = '%s_%s' % (elem['speaker'], sample)
        TO = os.path.join(OUT, DESTNAME)
        print(FROM, '=>', TO)
        copyfile(FROM, TO)