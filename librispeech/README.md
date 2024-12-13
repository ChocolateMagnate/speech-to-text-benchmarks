# LibriSpeech Dataset Processing Pipeline
This directory contains scripts for processing the LibriSpeech dataset, specifically focused on the dev-clean subset. The pipeline converts and combines multiple audio files into a single WAV file for benchmarking purposes.
# Dataset Overview
LibriSpeech is a large-scale corpus of read English speech, derived from audiobooks that are part of the LibriVox project. The dataset contains approximately 1000 hours of speech sampled at 16kHz, labeled and split into multiple subsets.
The dev-clean subset used in this pipeline contains:

* High-quality recordings from the development set
* Clear speech with minimal background noise
* Native English speakers with various accents
* Multiple speakers of different genders

# Pipeline Components
## combine-librispeech-dataset.sh

* Concatenates all audio files from the `dev-clean` subset recursively
* Concatenates all transcripts recursively
* Generates the final benchmark.wav file

## clean-librispeech-dataset.sh

* Cleans the ids from transcript file

# Output
The pipeline produces 2 output file:
```
audio/benchmark.wav
```
This file contains:
* Format: WAV (PCM)
* Sample rate: 16kHz
* Bit depth: 16-bit
* Channels: Mono
* Duration: ~5 hours

```
cleaned_combined_transcript.txt
```
This file is is the reference transcription for the entire output file. It's appropriate 
to use for WER (word error rate) tasks.

# Usage
Download preferred subset from [the official page](https://www.openslr.org/12).

Navigate to the unpacked directory and concatenate all files:
```commandline
bash combine-librispeech-dataset.sh
```
Then clean the transcript file:

```commandline
bash clean-librispeech-dataset.sh combined_transcript.txt
```

# Prerequisites
1. bash shell
2. ffmpeg
3. Sufficient storage space (~1GB for dev-clean subset)

# License
LibriSpeech is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
Citation
If you use this dataset in your research, please cite:
```citation
Copy@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```