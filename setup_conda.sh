#!/bin/bash

conda create -n SPDR python=3.5.2
source activate SPDR

pip install --ignore-installed --upgrade tensorflow
conda install numpy scipy theano=0.9.0 scikit-learn
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
conda install -c conda-forge keras=2.0.6
conda install -c conda-forge librosa=0.5.1
pip install https://github.com/dnouri/nolearn/archive/master.zip#egg=nolearn
conda install pandas=0.20.3
conda install -c omnia munkres=1.0.7
conda install matplotlib=2.1.0
pip install pydub
pip install auditok
conda install gcc
pip install pyannote.metrics
pip install pyyaml
pip install webrtcvad
pip install pysoundfile