from pyannote.core import Segment, Timeline, Annotation
from pyannote.core import notebook
from spdr.metrics import SPDR_Metrics
import matplotlib.pyplot as plt

reference = Annotation()
reference[Segment(0, 3)] = 'A'
reference[Segment(3, 5)] = 'B'
reference[Segment(6, 8)] = 'A'
reference[Segment(8, 10)] = 'B'
reference[Segment(12, 13)] = 'B'
reference[Segment(15, 18)] = 'A'
reference[Segment(18, 20)] = 'B'

hypothesis1 = Annotation()
hypothesis1[Segment(0, 3)] = 'a'
hypothesis1[Segment(3, 5)] = 'b'
hypothesis1[Segment(6, 8)] = 'a'
hypothesis1[Segment(8, 10)] = 'b'
hypothesis1[Segment(12, 13)] = 'b'
hypothesis1[Segment(15, 18)] = 'a'
hypothesis1[Segment(18, 20)] = 'b'

hypothesis2 = Annotation()
hypothesis2[Segment(0, 1)] = 'a'
hypothesis2[Segment(1, 2)] = 'c'
hypothesis2[Segment(2, 4)] = 'b'
hypothesis2[Segment(4, 5)] = 'd'
hypothesis2[Segment(6, 7)] = 'b'
hypothesis2[Segment(7, 8)] = 'c'
hypothesis2[Segment(8, 10)] = 'd'
hypothesis2[Segment(12, 13)] = 'b'
hypothesis2[Segment(15, 17)] = 'c'
hypothesis2[Segment(17, 18)] = 'e'
hypothesis2[Segment(18, 20)] = 'c'

hypothesis3 = Annotation()
hypothesis3[Segment(0, 5)] = 'a'
hypothesis3[Segment(6, 10)] = 'a'
hypothesis3[Segment(12, 13)] = 'a'
hypothesis3[Segment(15, 20)] = 'a'

uem = Timeline([Segment(float(0), float(20))])
spdr_metrics1 = SPDR_Metrics(reference, hypothesis1, uem)
spdr_metrics2 = SPDR_Metrics(reference, hypothesis2, uem)
spdr_metrics3 = SPDR_Metrics(reference, hypothesis3, uem)

text1 = "\nDER: %.4f | Purity: %.4f | Coverage: %.4f | PC-F-Score: %.4f" % (spdr_metrics1.get_DiarizationErrorRate(detailed=False), spdr_metrics1.get_DiarizationPurity(detailed=False), spdr_metrics1.get_DiarizationCoverage(detailed=False), spdr_metrics1.get_DiarizationCoveragePurityFScore(detailed=False))
text2 = "\nDER: %.4f | Purity: %.4f | Coverage: %.4f | PC-F-Score: %.4f" % (spdr_metrics2.get_DiarizationErrorRate(detailed=False), spdr_metrics2.get_DiarizationPurity(detailed=False), spdr_metrics2.get_DiarizationCoverage(detailed=False), spdr_metrics2.get_DiarizationCoveragePurityFScore(detailed=False))
text3 = "\nDER: %.4f | Purity: %.4f | Coverage: %.4f | PC-F-Score: %.4f" % (spdr_metrics3.get_DiarizationErrorRate(detailed=False), spdr_metrics3.get_DiarizationPurity(detailed=False), spdr_metrics3.get_DiarizationCoverage(detailed=False), spdr_metrics3.get_DiarizationCoveragePurityFScore(detailed=False))

# plot reference
plt.rcParams['figure.figsize'] = (10, 3)
plt.subplot(111)
notebook.plot_annotation(reference, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Reference', fontsize=12)
plt.xlabel('')
plt.savefig("./data/plots/thesis_metric/reference.png")
plt.close()

# plot hypothesis 1
plt.rcParams['figure.figsize'] = (10, 7)
plt.subplot(211)
notebook.plot_annotation(reference, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Reference', fontsize=12)
plt.xlabel('')
plt.subplot(212)
notebook.plot_annotation(hypothesis1, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Hypothesis perfect' + text1, fontsize=12)
plt.xlabel('')
plt.savefig("./data/plots/thesis_metric/hypothesis_correct.png")
plt.close()

# plot hypothesis 2
plt.rcParams['figure.figsize'] = (10, 7)
plt.subplot(211)
notebook.plot_annotation(reference, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Reference', fontsize=12)
plt.xlabel('')
plt.subplot(212)
notebook.plot_annotation(hypothesis2, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Hypothesis over-clustered' + text2, fontsize=12)
plt.xlabel('')
plt.savefig("./data/plots/thesis_metric/hypothesis_over_clustered.png")
plt.close()

 # plot hypothesis 3
plt.rcParams['figure.figsize'] = (10, 7)
plt.subplot(211)
notebook.plot_annotation(reference, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Reference', fontsize=12)
plt.xlabel('')
plt.subplot(212)
notebook.plot_annotation(hypothesis3, legend=True, time=True)
plt.gca().text(0.6, 0.15, 'Hypothesis under-clustered' + text3, fontsize=12)
plt.xlabel('')
plt.savefig("./data/plots/thesis_metric/hypothesis_under_clustered.png")
plt.close()

