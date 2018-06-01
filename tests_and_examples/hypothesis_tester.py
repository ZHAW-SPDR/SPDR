import pickle
from collections import Counter
from spdr.parser.hypothesis_generator import define_hypothesis
from spdr.metrics import SPDR_Metrics
from pyannote.core import Timeline


with open('dump.pkl', 'rb') as pFile:
    data = pickle.load(pFile)

sequences = data['sequences']
cluster_results = data['cluster_results']
uem = data['uem']
print(uem.for_json())
uem = Timeline().from_json({'pyannote': 'Timeline', 'content': [{'end': 1160.0, 'start': 1144.0}]})
#print(Counter(cluster_results))
h = define_hypothesis(sequences, cluster_results, uem)
spdr_metrics = SPDR_Metrics(h, h, uem)
spdr_metrics.get_Plot(80)