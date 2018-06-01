#!/bin/bash

for filename in data/in/RT09/**/*.sph; do
  sox -t sph $filename -r 16000 -c 1 -t wav ${filename%.*}.wav
  echo "converted $filename to ${filename%.*}.wav"
done