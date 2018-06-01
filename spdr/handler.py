"""
Dataset handler

This handler converts the RT-09 dataset to the correct format.
Also it parses the corresponding groundtruth files.
The parser is able to handle various eval tasks.

Requirements:
    * sph2pipe has to be installed in the path
        https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz

Usage: Use in code
    > handler = SPDR_RT09_Handler()
    > elements = handler.run()
    > handler.do_cleanup() # can be used to cleanup the generated wav files after processing has been done

Returns: An array of elements with the structure as follows
    [
        {
            'id': this is the ID of the recording it corresponds to the folders in config['data']['in']
            'channel': this is the amount of audio channels in the file (default is 1)
            'reference': the parsed and usable pyannote groundtruth
            'start': the start time (in s) of the scored snippet inside the audiofile 
            'end': the end time (in s) of the scored snippet inside the audiofile 
            'files': a list of files to be used (and present in audio in folder) for this task
        },
        ...
    ]

Dataset handler
optional arguments:
  -h, --help  show this help message and exit
  
"""
import shutil
import os
import subprocess
from pydub import AudioSegment
from pyannote.core import Annotation, Segment
from .utils import SPDR_Util


class SPDR_RT09_Handler():

    def __init__(self, ds=None):
        self.config = SPDR_Util.load_config()
        self.task = 'spkr'
        self.eval = self.config['spkr']['condition']
        self.data_in_base_dir = self.config['data']['in']
        self.dataset_original = self.config['data']['dataset_original']
        self.datasubset_to_use = self.config['data']['datasubset_to_use'] if ds is None else ds
        self.elements = []
        self.converted_file_format = 'wav'

    def run(self):
        if self.eval != 'MDM' and self.eval != 'SDM':
            raise Exception("Speaker diarization is only possible on SDM and MDM input condition")

        if self._check_dependencies():
            # parse uem file
            self._parse_uem()

            # parse audioList
            self._parse_audiolist()

            # convert files to correct format
            self._convert_files()

            # update the filehandle extensions to the newly created 
            self._update_file_extensions()

            # trim audio file to start and end
            # currently we do not trim because we would need to adjust the ground truth too
            # self._trim_audio_files()

            # load groundtruth
            self._parse_groundtruth()

            # return parsed elements
            return self.get_elements()
        else:
            raise ImportError("Check dependencies (sph2pipe is required)!")

    def get_elements(self):
        return self.elements

    @staticmethod
    def _check_dependencies():
        return shutil.which("sox")

    def _adjust_time(self, time):
        return time if not self.config['hypothesis']['scaledown'] else time / 1000

    def _fetch_file_in_folder(self, endswith, contains='', folder=None):
        if folder is None:
            folder = self.data_in_base_dir
        f = None
        for path, dirs, files in os.walk(folder):
            counter = 0
            for filename in files:
                if filename.endswith(endswith) and filename.find(contains) != -1:
                    counter += 1
                    if counter > 1:
                        raise Exception(
                            "Too many files ending with %s and containing '%s' present! Maybe you are trying to run different tasks at once - do not do that!" % (
                            endswith, contains))
                    f = os.path.join(path, filename)
        return f

    def _parse_uem(self):
        lines = [line.strip() for line in
                 open(self._fetch_file_in_folder(endswith='.uem', contains=self.eval.lower()), 'r').readlines()]
        for line in lines:
            if not line.startswith(';'):
                entry = line.split(' ')
                if entry[0] == self.datasubset_to_use:  # only work on the dataset selected
                    self.elements.append({
                        'id': entry[0],
                        'channel': entry[1],
                        'start': self._adjust_time(float(entry[2])),
                        'end': self._adjust_time(float(entry[3]))
                    })

    def _parse_audiolist(self):
        lines = [line.strip() for line in
                 open(self._fetch_file_in_folder(endswith='audioList.txt', contains=self.eval.lower()),
                      'r').readlines()]
        for line in lines:
            if not line.startswith(';'):
                entry = line.split(' ')
                identifier = entry[0]
                files = [e.replace(self.dataset_original, self.data_in_base_dir) for e in entry[1:len(entry)]]
                for element in self.elements:
                    if element['id'] == identifier:
                        element['files'] = files

    @staticmethod
    def convert_file(file_handle, file_format):
        converted_file_handle = file_handle.replace('sph', file_format)

        print('Converting File %s' % file_handle)
        bash_command = ['sox', '-t', 'sph', file_handle, '-r', '16000', '-c', '1', '-t',
                        file_format, converted_file_handle]

        process = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error is not None:
            raise Exception("An error occured during conversion (%s): %s" % (output, error))

    def _convert_files(self):
        for element in self.elements:
            for file_handle in element['files']:
                converted_file_handle = file_handle.replace('sph', self.converted_file_format)
                if os.path.isfile(converted_file_handle):
                    continue

                SPDR_RT09_Handler.convert_file(file_handle, self.converted_file_format)

    def _update_file_extensions(self):
        for i, element in enumerate(self.elements):
            for j, filehandle in enumerate(element['files']):
                self.elements[i]['files'][j] = filehandle.replace('sph', self.converted_file_format)

    def _trim_audio_files(self):
        for element in self.elements:
            start = int(float(element['start']) * 1000)
            end = int(float(element['end']) * 1000)
            for filehandle in element['files']:
                audio = AudioSegment.from_wav(filehandle)
                relevant_snippet = audio[start:end]
                relevant_snippet.export(filehandle, format=self.converted_file_format)

    def _parse_groundtruth(self):
        for element in self.elements:
            groundtruthhandle = os.path.join(self.data_in_base_dir, element['id'], element['id'] + '.txt')
            lines = [line.strip() for line in open(groundtruthhandle, 'r').readlines()]
            reference = Annotation(uri=element['id'])
            speakers = []

            for line in lines:
                # check whether tab or space delimited
                entry = line.split(' ')
                if len(entry) < 3:
                    entry = line.split('\t')
                ref_start = self._adjust_time(float(entry[0]))
                ref_end = self._adjust_time(float(entry[1]))
                ref_spkr = (entry[2]).replace(':', '')
                reference[Segment(ref_start, ref_end)] = ref_spkr
                speakers.append(ref_spkr)

            element['reference'] = reference
            element['speakers'] = set(speakers)

    def do_cleanup(self):
        if self.elements is not None or len(self.elements) >= 1:
            for i, element in enumerate(self.elements):
                for j, filehandle in enumerate(element['files']):
                    if filehandle.endswith(self.converted_file_format):
                        bashCommand = ['rm', filehandle]
                        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
                        output, error = process.communicate()

                        if error is not None:
                            raise Exception("An error occured during deletion (%s): %s" % (output, error))
