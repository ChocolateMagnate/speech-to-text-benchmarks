# Speech-to-Text Model Benchmarking
This script benchmarks different variants of OpenAI's Whisper speech recognition model by measuring their Word Error Rate (WER) and Real-Time Factor (RTF) on audio transcription tasks.

# Overview
The script processes an audio file using a specified Whisper model variant, measures the transcription time, and compares the output against a reference transcription to calculate accuracy metrics.
# Requirements

* Python 3.x (tested on Python 3.12)
* PyTorch
* Transformers library
* jiwer (for WER calculation)
* scipy (for audio processing)
* CUDA-capable GPU (optional but recommended)

# Quick start
```shell
# 1. Install dependencies
poetry install
# 2. Activate virtual environment
source $(poetry env info --path)/bin/activate
# 3. Run benchmark
python transcribe.py
```

# Usage
```commandline
python transcribe.py [-h] [-a AUDIO] [-o OUTPUT] [-s SOURCE] [-v {large-v3-turbo,medium,base,tiny}]
```
Arguments

-a, --audio: Path to input audio file (default: "audio/benchmark.wav")

-o, --output: Path for saving transcription output (default: "transcriptions/transcription:latest.txt")

-s, --source: Path to reference transcription for WER calculation (default: "source-transcriptions/transcription.txt")

-v, --variant: Whisper model variant to use (default: "tiny")

# Methodology

Input: 5-hour audio recording with corresponding reference transcription

Models: Tested four Whisper variants (tiny, base, medium, large-v3-turbo)

Metrics:

1. RTF (Real-Time Factor) = Processing Time / Audio Duration
2. WER (Word Error Rate) compared against reference transcription

# Benchmark Results
Model VariantRTFWERProcessing Time (s)tiny0.011525.37%223.08base0.015623.14%301.96turbo0.020320.12%394.44medium0.049621.62%961.93
Analysis

All models achieved faster-than-real-time processing (RTF < 1.0)

Larger models generally showed better accuracy (lower WER)

The large-v3-turbo variant achieved the best accuracy while maintaining reasonable speed

The tiny model offered the fastest processing with acceptable accuracy for many use cases

| Whisper variant | RTF      | WER    |
|-----------------|----------|--------|
| tiny            | 0.0115   | 0.2536 |
| base            | 0.0155   | 0.2313 |
| medium          | 0.04959  | 0.2162 |
| large-v3-turbo  | 0.02033  | 0.2012 |

# Implementation Details
The script handles:

* Audio file loading and preprocessing
* Resampling to 16kHz if needed
* Mono channel conversion for multi-channel audio
* CUDA acceleration when available
* Comprehensive logging of process metrics

## Audio origin
The `audio/benchmark.wav` file was obtained from LibriSpeech `dev-clean` subset.
It was composed by concatenating multiple .flac files and converting them into 
a single .wav file using ffmpeg(1). For more implementation details, see `librispeech/README.md`.

# Conclusion
The `openai/whisper-large-v3-turbo` model offers the most value for enterprise STT by 
delivering state-of-the-art latency and accuracy. Although competitive, I consider Deepgram
to be a better pick for business because of its superior WER in the Nova model (5-15% WER).
Investing into Whisper also comes with hidden costs including, but not limiting to:
* Hiring and paying DevOps for deployment
* Ongoing cloud costs
* Manual configuration, etc.

Deepgram Nova offers an advantage of better accuracy and simplicity of use, which positions it
in better consideration than OpenAI Whisper. Therefore, I would pick Deepgram for its reliability,
superior performance and enterprise support over Whisper. The only advantages Whisper has that would
make it a better option would either involve the need to build a new model (using Whisper as the base)
or one-time offline transcriptions (where latency does not matter as well).

# Notes

Processing times may vary based on hardware configuration.

GPU acceleration significantly improves processing speed.

Memory requirements increase with model size.