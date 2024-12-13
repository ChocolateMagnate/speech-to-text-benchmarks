#!/usr/bin/env python3
# transcribe.py -- Benchmark Whisper transcriptions RTF and WER
import sys
import time
import wave
import signal
import logging
import argparse
import warnings
from pathlib import Path

import jiwer
import scipy
import torch
import numpy as np
from transformers import pipeline
from numpy.typing import NDArray

NORMALIZATION_RATE = 2 ** 15

# HuggingFace Whisper implementations uses a deprecated inputs field that is being updated.
# We silence the warnings in order to maintain clean logger output because this warning is not our responsibility.
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")

logger = logging.getLogger("transcribe.py")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(f"transcriber.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def flush_logs_on_interrupt(sig, frame) -> None:
    logging.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, flush_logs_on_interrupt)


class Whisper:
    def __init__(self, huggingface_model_name: str, sliding_window_context_seconds: int = 10,
                 stride_seconds: int = 1, confidence_threshold: float = 0.5):
        self.huggingface_model_name = huggingface_model_name
        self.sliding_window_context_seconds = sliding_window_context_seconds
        self.confidence_threshold = confidence_threshold
        self.stride_seconds = stride_seconds

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device chosen: {self.device}")
        whisper_loading_start = time.time()
        self.model = pipeline("automatic-speech-recognition",
                              model=huggingface_model_name,
                              device=self.device)
        whisper_loading_end = time.time()
        whisper_loading_time = whisper_loading_end - whisper_loading_start
        logger.info(f"Loaded {self.huggingface_model_name} in {whisper_loading_time} seconds.")

        self.output_path = Path("transcription.txt")
        self.total_processing_time = 0.0
        self.max_processing_time = 0.0
        self.chunks_processed = 0
        self.sample_rate = 16000  # Whisper expects 16kHz sampling rate
        self.current_time = 0.0
        self.previous_text = ""

    def load_wav(self, audio_path: Path) -> tuple[float, NDArray]:
        wav_loading_start = time.time()
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            original_frame_rate = wav.getframerate()

            duration = wav.getnframes() / original_frame_rate
            audio_data = audio_data.astype(np.float32) / NORMALIZATION_RATE

            if wav.getnchannels() > 1:  # Handle changing to mono channel
                audio_data = audio_data.reshape(-1, wav.getnchannels())
                audio_data = audio_data.mean(axis=1)

            if original_frame_rate != self.sample_rate:  # Handle resampling
                logger.info(f"Resampling {audio_path} from {original_frame_rate}Hz to {self.sample_rate}Hz.")
                number_of_samples = int(len(audio_data) * self.sample_rate / original_frame_rate)
                audio_data = scipy.signal.resample(audio_data, number_of_samples)

        wav_loading_end = time.time()
        logger.info(f"Loaded {audio_path} in {wav_loading_end - wav_loading_start} second(s).")
        return duration, audio_data

    @torch.no_grad()
    def transcribe(self, audio_path: Path) -> tuple[float, str]:
        duration, audio = self.load_wav(audio_path)
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        logger.info(f"Started transcribing {audio_path} of {hours} hours and {minutes} minutes...")
        result = self.model(audio, return_timestamps=True)
        return duration, result["text"]


def main():
    parser = argparse.ArgumentParser("transcribe.py")
    parser.add_argument("-a", "--audio", action="store", dest="audio",
                        default=Path("audio/benchmark.wav"),
                        help="The path to the audio file to transcribe.")
    parser.add_argument("-o", "--output", action="store", dest="output",
                        default=Path("transcriptions/transcription:latest.txt"),
                        help="The path where the transcription must be saved.")
    parser.add_argument("-s", "--source", action="store", dest="source",
                        default=Path("source-transcriptions/transcription.txt"),
                        help="The path to the source of truth transcription to use for WER calculation.")
    parser.add_argument("-v", "--variant", action="store", dest="variant",
                        choices=["large-v3-turbo", "medium", "base", "tiny"], default="tiny",
                        help="The Whisper model variant to use.")
    arguments = parser.parse_args()
    huggingface_model_name = f"openai/whisper-{arguments.variant}"

    model = Whisper(huggingface_model_name)

    transcription_start = time.time()
    wav_duration, hypothesis = model.transcribe(arguments.audio)
    transcription_end = time.time()
    transcription_duration = transcription_end - transcription_start
    rtf = transcription_duration / wav_duration

    with open(arguments.output, "w") as output_file:
        output_file.write("".join(hypothesis))

    with open(arguments.source, "r") as source:
        reference = source.read()

    wer = jiwer.wer(reference.lower(), hypothesis.lower())
    logger.info(f"Processed {wav_duration}s in {transcription_duration} ({rtf} RTF) with {wer} WER.")
    logger.info(f"Transcription saved to: {arguments.output}")
    logger.info(f"Transcribed file: {arguments.audio}")
    logger.info(f"The ground truth transcription file: {arguments.source}")


if __name__ == "__main__":
    main()
