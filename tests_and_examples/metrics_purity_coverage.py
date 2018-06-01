from pyannote.core import Segment, Timeline, Annotation
from pyannote.core import notebook
from spdr.metrics import SPDR_Metrics
import matplotlib.pyplot as plt
import numpy as np

reference = Annotation()
reference[Segment(0, 6)] = 'A'
reference[Segment(6, 12)] = 'B'
reference[Segment(12, 20)] = 'C'


hypothesis1 = Annotation()
hypothesis1[Segment(0, 10)] = 'a'
hypothesis1[Segment(10, 13)] = 'b'
hypothesis1[Segment(13, 20)] = 'c'


purity = Annotation()
purity[Segment(0,6)] = 'A'
purity[Segment(10,12)] = 'A'
purity[Segment(13,20)] = 'A'

coverage = Annotation()
coverage[Segment(0,6)] = 'B'
coverage[Segment(6,10)] = 'B'
coverage[Segment(13,20)] = 'B'


precisionrecall = Annotation()


uem = Timeline([Segment(float(0), float(20))])
spdr_metrics = SPDR_Metrics(reference, hypothesis1, uem)

text1 = "\nDER: %.4f | Purity: %.4f | Coverage: %.4f | PC-F-Score: %.4f" % (\
    spdr_metrics.get_DiarizationErrorRate(detailed=False), \
    spdr_metrics.get_DiarizationPurity(detailed=False), \
    spdr_metrics.get_DiarizationCoverage(detailed=False), \
    spdr_metrics.get_DiarizationCoveragePurityFScore(detailed=False), \
     )

# plot reference
plt.rcParams['figure.figsize'] = (10, 9)
plt.subplot(311)
notebook.plot_annotation(reference, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Reference', fontsize=12, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
plt.xticks(np.linspace(0,20,21))
plt.xlabel('')
plt.grid()

# plot hypothesis 
plt.subplot(312)
notebook.plot_annotation(hypothesis1, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Hypothesis' + text1, fontsize=12, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
plt.xticks(np.linspace(0,20,21))
plt.xlabel('')
plt.grid()

# plot purity
plt.subplot(313)
notebook.plot_annotation(purity, legend=False, time=True)
plt.gca().text(0.6, 0.15, 'Purity', fontsize=12, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
plt.xticks(np.linspace(0,20,21))
plt.xlabel('')
plt.grid()

plt.savefig("./data/plots/thesis_metric_purity_coverage/purity.png")
plt.close()

# plot coverage

# plot reference
plt.rcParams['figure.figsize'] = (10, 9)
plt.subplot(311)
notebook.plot_annotation(reference, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Reference', fontsize=12, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
plt.xticks(np.linspace(0,20,21))
plt.xlabel('')
plt.grid()

# plot hypothesis 
plt.subplot(312)
notebook.plot_annotation(hypothesis1, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Hypothesis' + text1, fontsize=12, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
plt.xticks(np.linspace(0,20,21))
plt.xlabel('')
plt.grid()

plt.subplot(313)
notebook.plot_annotation(coverage, legend=False, time=True)
plt.gca().text(0.6, 0.15, 'Coverage', fontsize=12, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
plt.xticks(np.linspace(0,20,21))
plt.xlabel('')
plt.grid()

plt.savefig("./data/plots/thesis_metric_purity_coverage/coverage.png")
plt.close()
