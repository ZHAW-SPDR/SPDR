import os
import argparse
from pydub import AudioSegment
from .utils import get_filename_without_extension


def _fixed_size_segmenter(filename, segment_path, duration):
    segment_folder = os.path.join(segment_path, get_filename_without_extension(os.path.basename(filename)))
    audio = AudioSegment.from_wav(filename)
    slices = audio[::duration]

    _prepare_segment_folder(segment_folder)

    for i, s in enumerate(slices):
        s.export(os.path.join(segment_folder, str(i) + ".wav"), format="wav")


def _prepare_segment_folder(segment_folder):
    if not os.path.exists(segment_folder):
        os.mkdir(segment_folder)
    else:
        _clear_dir(segment_folder)


def _fixed_size_segmenter_from_to(filename, segment_path, duration, timeline):
    segment_folder = os.path.join(segment_path, get_filename_without_extension(os.path.basename(filename)))
    audio = AudioSegment.from_wav(filename)

    slices = audio[timeline[0]:timeline[1]:duration]

    _prepare_segment_folder(segment_folder)

    for i, s in enumerate(slices):
        s.export(os.path.join(segment_folder, str(i) + ".wav"), format="wav")


def _wave_files(path):
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith((".wav", ".WAV")) and os.path.isfile(os.path.join(root, filename)):
                yield os.path.join(root, filename)


def _clear_dir(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def segment_wave_files_from_to(files, segment_path, timeline, segment_size=1000):
    for wave_file in files:
        _fixed_size_segmenter_from_to(wave_file, segment_path, segment_size, timeline=timeline)


def segment_wave_files(files, segment_path, segment_size=1000):
    for wave_file in files:
        _fixed_size_segmenter(wave_file, segment_path, segment_size)


def segment_files_in_path(path, segment_path, segment_size=1000):
    for wave_file in _wave_files(path):
        _fixed_size_segmenter(wave_file, segment_path, segment_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="audio file segmenter")
    parser.add_argument("path", type=str, help="path to the the wave-files")
    parser.add_argument("segment_path", type=str, help="destination path for generated segments")
    parser.add_argument("segment_size", type=int, help="segment duration in ms")

    args = parser.parse_args()
    segment_files_in_path(path=args.path, segment_path=args.segment_path, segment_size=args.segment_size)
