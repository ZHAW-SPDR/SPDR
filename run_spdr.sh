#!/bin/bash

CUR_DIR="$(pwd)"
PYTHONPATH=$CUR_DIR:$CUR_DIR/ZHAW_deep_voice:$PYTHONPATH
export PYTHONPATH

KERAS_BACKEND="tensorflow"
export KERAS_BACKEND

python controller.py