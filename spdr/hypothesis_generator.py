from pyannote.core import Annotation, Segment
from .utils import SPDR_Util


def define_hypothesis(sequences, cluster_mapping):
    hypothesis = Annotation()
    config = SPDR_Util.load_config()
    scaledown = config['hypothesis']['scaledown']
    i = 0

    while i < len(sequences):
        start = sequences[i].start
        end = sequences[i].end

        while i < len(sequences) - 1 and cluster_mapping[i] == cluster_mapping[i + 1]:
            end = sequences[i + 1].end
            i += 1
        start = start if not scaledown else start / 1000
        end = end if not scaledown else end / 1000
        hypothesis[Segment(float(start), float(end))] = cluster_mapping[i]
        i += 1

    return hypothesis


def define_hypothesis_for_embeddings(cluster_mapping, start_time=0):
    hypothesis = Annotation()
    config = SPDR_Util.load_config()
    segment_size = config['segment']['size'] / 1000

    i = 0
    start = 0.0

    while i < len(cluster_mapping):
        start = start_time + (i * segment_size)
        end = start + segment_size

        if cluster_mapping[i] >= 0:
            while i < len(cluster_mapping) - 1 and cluster_mapping[i] == cluster_mapping[i + 1]:
                end += segment_size
                i += 1

            hypothesis[Segment(start, end)] = cluster_mapping[i]

        i += 1

    return hypothesis
