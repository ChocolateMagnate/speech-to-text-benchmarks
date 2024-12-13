#!/bin/bash
# combine-librispeech-dataset.sh -- Concatenate files in a single file

# Create a temporary file listing all audio files
find . -name "*.flac" | sort > audio_files.txt

# Convert the file list to FFmpeg concat format
while read -r line; do
    echo "file '$line'" >> concat_list.txt
done < audio_files.txt

# Concatenate all audio files using FFmpeg
ffmpeg -f concat -safe 0 -i concat_list.txt -c:a flac combined_audio.flac

# Convert .flac files into .wav
ffmpeg -i combined_audio.flac -c:a pcm_s16le benchmark.wav

# Combine all transcripts
find . -name "*.trans.txt" -exec cat {} \; | sort > combined_transcript.txt

# Clean up temporary files
rm audio_files.txt concat_list.txt combined_audio.flac

echo "Audio files have been concatenated to combined_audio.flac"
echo "Transcripts have been combined into combined_transcript.txt"

# Show duration of final file
duration=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 combined_audio.flac)
echo "Total duration: $duration seconds"
