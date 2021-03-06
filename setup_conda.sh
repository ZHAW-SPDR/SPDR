#!/bin/bash

conda create -n SPDR python=3.5.2
source activate SPDR

pip install --ignore-installed --upgrade tensorflow
conda install -y numpy scipy theano=0.9.0 scikit-learn
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
conda install -cy conda-forge keras=2.0.6
conda install -cy conda-forge librosa=0.5.1
pip install https://github.com/dnouri/nolearn/archive/master.zip#egg=nolearn
conda install -y pandas=0.20.3
conda install -cy omnia munkres=1.0.7
conda install matplotlib=2.1.0
pip install pydub
pip install auditok
conda install -y gcc
pip install pyannote.metrics
pip install pyyaml
pip install webrtcvad
pip install pysoundfile
pip install hdbscan