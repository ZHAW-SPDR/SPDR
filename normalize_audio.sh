#!/bin/bash

ffmpeg_normalize="ffmpeg-normalize"

if ! type $ffmpeg_normalize > /dev/null; then
    pip install ffmpeg-normalize
fi

for filename in data/in/RT09/**/*.wav; do
  ffmpeg-normalize $filename -ar 16000 -o ${filename%.*}_normalized.wav -f
  echo "normalized $filename. normalized file stored under ${filename%.*}_normalized.wav"
done