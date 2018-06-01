"""
Abstract parser for parsing ground truth for speaker diarization.

A derived parser should implement the 'parse()' function, which 
will be called once the parser has been initialized.
!!! IMPORTANT !!!
The self.parsed flag has to be set to true after parsing in order to
retrieve the reference data. 'get_Reference()' can and will raise an
exception, if the flag is not set.

The reference data is a dictionary with the following structure:
dict(
    'FILENAME1': pyannote.core.Annotation(),
    'FILENAME2': pyannote.core.Annotation(),
    ...
)
(FILENAME = is the audio file to be analyzed)

Arguments:
    basedir -- The basedir for the annotation data

Usage: Don't instantiate this class directly.

Abstract parser for parsing ground truth for Speaker diarization
optional arguments:
  -h, --help  show this help message and exit
  
"""
from abc import ABC, abstractmethod

class Parser(ABC):
    def __init__(self, basedir):
        self.reference = dict()
        self.basedir = basedir
        self.parsed = False
    
    @abstractmethod
    def parse(self):
        pass

    def get_Reference(self):
        if self.parsed:
            return self.reference
        else:
            raise Exception('Ground thruth has not been parsed. Run Parser.parse() first.')