from spdr.metrics import SPDR_Metrics
from pyannote.core import Segment, Timeline, Annotation

# get metrics
reference = Annotation(uri='file2')
reference[Segment(0, 5)] = 'A'
reference[Segment(6, 10)] = 'B'
reference[Segment(12, 13)] = 'B'
reference[Segment(15, 20)] = 'A'

hypothesis = Annotation(uri='file2')
hypothesis[Segment(0, 5)] = 'a'
hypothesis[Segment(6, 10)] = 'b'
hypothesis[Segment(12, 13)] = 'b'
hypothesis[Segment(15, 20)] = 'a'

uem = Timeline([Segment(float(6), float(12))])
spdr_metrics = SPDR_Metrics(reference, hypothesis, uem)
print("DER simple")
print(spdr_metrics.get_DiarizationErrorRate(detailed=False))
print("DER detailed")
print(spdr_metrics.get_DiarizationErrorRate(detailed=True))
print("DER simple greedy")
print(spdr_metrics.get_DiarizationErrorRate(greedy=True, detailed=False))
print("DER detailed greedy")
print(spdr_metrics.get_DiarizationErrorRate(greedy=True, detailed=True))
spdr_metrics.get_Plot()

print("")

