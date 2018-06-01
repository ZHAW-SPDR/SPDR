"""
Metrics for speaker diarization.

This is wrapper class for the most common metrics used around speaker diarization.
The metrics are described in this paper:
http://herve.niderb.fr/download/pdfs/Bredin2017a.pdf

More detail can be found here: 
https://pyannote.github.io/pyannote-metrics/index.html

Usage: Use in code
    spdr_metrics = SPDR_Metrics(reference_data, hypothesis_data)

Metrics for Speaker diarization
optional arguments:
  -h, --help  show this help message and exit
  
"""

from pyannote.metrics.diarization import *
from pyannote.metrics.segmentation import *
from pyannote.core import notebook
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.core import Annotation
from pyannote.core import SlidingWindow
from pyannote.core import SlidingWindowFeature
import matplotlib.pyplot as plt

class SPDR_Metrics():

    def __init__(self, reference, hypothesis, uem):
        self.reference = reference
        self.hypothesis = hypothesis
        self.uem = uem
    
    def get_DiarizationErrorRate(self, greedy=False, detailed=False):
        """ 
            Get the diarization error rate for an object

            Parameters
            ----------
            greedy      : bool, optional
                Two implementations of the diarization error rate are available (optimal and greedy), 
                depending on how the one-to-one mapping between reference and hypothesized speakers is computed.

                The optimal version uses the Hungarian algorithm to compute the mapping that minimize the confusion term, 
                while the greedy version operates in a greedy manner, mapping reference and hypothesized speakers iteratively, 
                by decreasing value of their cooccurrence duration.

                In practice, the greedy version is much faster than the optimal one, 
                especially for files with a large number of speakers – though it may slightly over-estimate 
                the value of the diarization error rate.
            
            detailed    : bool, optional
                When true returns an object like this:
                {   'confusion': 7.0,
                    'correct': 22.0,
                    'diarization error rate': 0.5161290322580645,
                    'false alarm': 7.0,
                    'missed detection': 2.0,
                    'total': 31.0
                }
                When false, returns a the single DER value (in our example above 0.5161290322580645)
        """
        if greedy:
            metric = GreedyDiarizationErrorRate()
        else:
            metric = DiarizationErrorRate(skip_overlap=True)
        return metric(self.reference, self.hypothesis, detailed=detailed, uem=self.uem)
    
    def get_Plot(self, title=None, text=None, width=10, get_plt=False):
        """Plot the reference and hypothesis

        """
        notebook.width = width
        plt.figure(figsize=(notebook.width, 10))
        
        pTitle = ('Diarization Performance on %s' % title) if title is not None else 'Diarization Performance'
        plt.suptitle(pTitle)

        # plot reference
        plt.subplot(211)
        notebook.plot_annotation(self.reference, legend=True, time=True)
        plt.gca().text(4, 0.15, 'Reference', fontsize=10, bbox=dict(edgecolor='None', facecolor='white', alpha=1))

        # plot hypothesis
        ptext = ''
        if text is not None:
            ptext = '\n'+text
        plt.subplot(212)
        notebook.plot_annotation(self.hypothesis, legend=True, time=True)
        plt.gca().text(4, 0.1, 'Hypothesis' + ptext, fontsize=10, bbox=dict(edgecolor='None', facecolor='white', alpha=1))
        plt.subplots_adjust(hspace=0.4)
        if get_plt:
            return plt
        else:
            plt.show()

    def get_DiarizationPurity(self, weighted=True, detailed=False):
        """Cluster purity

        A hypothesized annotation has perfect purity if all of its labels overlap
        only segments which are members of a single reference label.

        Parameters
        ----------
        weighted    : bool, optional
            When True (default), each cluster is weighted by its overall duration.
        detailed    : bool, optional
        """
        metric = DiarizationPurity(weighted=weighted)
        detail = metric.compute_components(self.reference, self.hypothesis)
        if detailed:
            return detail
        else: 
            return metric.compute_metric(detail)

    def get_DiarizationCoverage(self, weighted=True, detailed=False):
        """Cluster coverage

        A hypothesized annotation has perfect coverage if all segments from a
        given reference label are clustered in the same cluster.

        Parameters
        ----------
        weighted    : bool, optional
            When True (default), each cluster is weighted by its overall duration.
        detailed    : bool, optional
        """
        metric = DiarizationCoverage(weighted=weighted)
        detail = metric.compute_components(self.reference, self.hypothesis)
        if detailed:
            return detail
        else: 
            return metric.compute_metric(detail)

    def get_DiarizationCoveragePurityFScore(self, detailed=False):
        metric = DiarizationPurityCoverageFMeasure()
        detail = metric.compute_components(self.reference, self.hypothesis)
        if detailed:
            return detail
        else: 
            return metric.compute_metric(detail)
    
    def get_DiarizationHomogenity(self, detailed=False):
        """Cluster homogeneity"""
        metric = DiarizationHomogeneity()
        detail = metric.compute_components(self.reference, self.hypothesis)
        if detailed:
            return detail
        else: 
            return metric.compute_metric(detail)

    def get_DiarizationCompleteness(self, detailed=False):
        """Cluster completeness"""
        metric = DiarizationCompleteness()
        detail = metric.compute_components(self.reference, self.hypothesis)
        if detailed:
            return detail
        else: 
            return metric.compute_metric(detail)
        
