# Speaker diarization suite

This is our speaker diarization suite used and developed in our bachelor thesis.

## Setup
This project depends on the ZHAW_deep_voice suite, which is included as git submodule.
To checkout the project with all submodules use:
~~~~
git pull && git submodule init && git submodule update && git submodule status
~~~~

To run SPDR a conda enviroment is required. Execute following script t create
a conda environment with all dependencies:
~~~~
$ ./setup_conda.sh
~~~~



## How to use
~~~~
usage: controller.py [-h] path segment_path segment_size

Controller for Speaker diarization

positional arguments:
  path          path to the the wave-files
  segment_path  destination path for generated segments
  segment_size  segment duration in ms

optional arguments:
  -h, --help    show this help message and exit
~~~~


## IDE Settings (PyCharm)

Add the SPDR conda environment (created via setup_conda.sh) according this documentation:
https://docs.anaconda.com/anaconda/user-guide/tasks/integration/pycharm.

Mark the folder "ZHAW_deep_voice" as "Source Root".