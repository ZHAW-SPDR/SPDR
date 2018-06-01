"""
The util class for the speaker diarization suite.
Usage: Use in code
    SPDR_Util().load_config()

The util class for the speaker diarization suite.
optional arguments:
  -h, --help  show this help message and exit
  
"""
from collections import namedtuple

import os
import yaml


class SPDR_Util():
    @staticmethod
    def load_config():
        with open("config.yml") as config:
            return yaml.safe_load(config)

    @staticmethod
    def load_config_by(config_file):
        with open(config_file) as config:
            return yaml.safe_load(config)


Sequence = namedtuple("Sequence", ["embedding", "start", "end"])


def get_filename_without_extension(filename):
    extension = os.path.splitext(filename)[1]
    return filename.replace(extension, "")


def get_filename_with_new_extension(filename, new_extension):
    extension = os.path.splitext(filename)[1]
    return os.path.basename(filename).replace(extension, new_extension)
