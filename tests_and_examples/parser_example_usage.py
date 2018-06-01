from spdr.metrics import SPDR_Metrics
from spdr.parser.voxceleb_parser import Voxceleb_Parser

vp = Voxceleb_Parser('./data/groundtruth/voxceleb1')
vp.parse()
ref = vp.get_Reference()

spdr_metrics = SPDR_Metrics(ref['_dyeYOti1_E'], ref['_dyeYOti1_E']) #second arg should be hypothesis
print(spdr_metrics.get_DiarizationErrorRate(detailed=True))
spdr_metrics.get_Plot()