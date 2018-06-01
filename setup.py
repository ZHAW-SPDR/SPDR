from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='spdr',
   version='1.0',
   description='BA stdm 02 - Speaker Diarization',
   license="MIT",
   long_description=long_description,
   author='Niclas Simmler, Amin Trabi',
   author_email='simmlnic@students.zhaw.ch;trabiami@students.zhaw.ch',
   url="https://www.zhaw.ch/",
   packages=['spdr'],
   install_requires=['librosa', 'auditok', 'pydub', 'numpy', 'scipy', 'ggplot']
)