# Speaker diarization suite

This is our speaker diarization suite used and developed in our bachelor thesis.

## System requirements
A linux or Mac OS is required to run the suite, where anaconda is installed.
An anaconda installer is available under: https://www.anaconda.com/download/

Additionally sox have to be installed, which is available under: http://sox.sourceforge.net/
On Ubuntu sox can be installed with apt:

~~~~
$ sudo apt-get install sox
~~~~

## Setup
This project depends on the ZHAW_deep_voice suite, which is included as git submodule.
To checkout the project with all submodules use:
~~~~
git pull && git submodule init && git submodule update && git submodule status
~~~~

To run SPDR a anaconda enviroment is required. Execute following script t create
a conda environment with all dependencies:
~~~~
$ ./setup_conda.sh
~~~~

Following dependencies will be installed:

- tensorflow
- numpy
- scipy
- theano=0.9.0
- Lasagne
- keras=2.0.6
- librosa=0.5.1
- nolearn
- pandas=0.20.3
- omnia
- munkres=1.0.7
- matplotlib=2.1.0
- pydub
- auditok
- pyannote.metrics
- pyyaml
- webrtcvad
- pysoundfile


## Datasets and corporas
The RT-09 corpora is required to use the suite. This dataset is provided by the National Institute of Standards and Technology (NIST).
Contact NIST to obtain the RT-09 dataset. The dataset have to be stored under ./data/in/RT09.


## How to use

If the environment "SPDR" is active, run following script within the project root directory:
~~~~
$ ./run_spdr.sh
~~~~

Activate the environment "SPDR" with following command:
~~~~
$ source activate SPDR
~~~~

## Development

### IDE Settings PyCharm

Add the SPDR conda environment (created via setup_conda.sh) according this documentation:
https://docs.anaconda.com/anaconda/user-guide/tasks/integration/pycharm.

Mark the folder "ZHAW_deep_voice" as "Source Root".