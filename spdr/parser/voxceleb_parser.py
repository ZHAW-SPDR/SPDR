"""
Parser for parsing ground truth in Voxceleb1 for speaker diarization.

The 'parse()' implementation is derived from here:
https://github.com/pyannote/pyannote-db-voxceleb/blob/master/scripts/prepare_data.ipynb

Multiple speaker files per audio track can be put into the basedir.

Arguments:
    basedir -- The basedir for the annotation data

Usage:
    vp = Voxceleb_Parser('./data/groundtruth/voxceleb1')
    vp.parse()
    ref = vp.get_Reference()

Parser for parsing ground truth in Voxceleb1 for speaker diarization.
optional arguments:
  -h, --help  show this help message and exit
  
"""
from .parser import Parser
from pandas import DataFrame as df
from glob import glob
from tqdm import tqdm
from pyannote.core import Annotation, Segment

class Voxceleb_Parser(Parser):
    def __init__(self, basedir):
        super().__init__(basedir)
    
    def parse(self):
        glob_exp = '{voxceleb_dir}/*/*.txt'.format(voxceleb_dir=self.basedir)
        segments = []
        for path_txt in tqdm(glob(glob_exp)):
            segments.extend(list(self._extract_information(path_txt)))
        data = df(segments)
        data.set_index('segment', inplace=True)
        columns = ['audiofile', 'end', 'speaker', 'start', 'uri', 'verification']
        data.columns = columns
        self._gen_Reference(data)
        self.parsed = True
    
    def _extract_information(self, txt):
        lines = [line.strip() for line in open(txt, 'r').readlines()]
        speaker = lines[0].split('\t')[-1]
        uri = lines[1].split('\t')[-1]
        subset = lines[3].split('\t')[-1]
        if subset == 'test':
            subset = 'tst'
        for line in lines[5:]:
            segment, start, end = line.split()
            yield {'speaker': speaker, 
                'uri': speaker + '/' + uri,
                'audiofile': uri, 
                'start': float(start), 
                'end': float(end), 
                'segment': segment, 
                'verification': subset}
    
    def _gen_Reference(self, data):
        for af in data['audiofile'].unique():
            print(af)
            ref = Annotation()
            for index,row in data.loc[data['audiofile'] == af].iterrows():
                ref[Segment(row['start'], row['end'])] = row['speaker']
            self.reference[af] = ref