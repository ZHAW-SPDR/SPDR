from pyannote.core import Segment


def plot_embeddings(embeddings, reference, timeline, segment_size):
    embedding_mapping = []
    for i, embedding in enumerate(embeddings):
        speaker = reference[Segment(start=timeline[0] + i * segment_size)]
        embedding_mapping.append(speaker)

