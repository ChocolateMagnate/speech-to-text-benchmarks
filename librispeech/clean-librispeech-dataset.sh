#!/bin/bash
# clean-librispeech-dataset.sh --

# Check if input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 input_transcript.txt"
    exit 1
fi

input_file="$1"
output_file="cleaned_${input_file}"

# Read each line, remove the ID, and write to new file
while IFS= read -r line; do
    # Cut off the first word (ID) and print the rest
    echo "$line" | cut -d' ' -f2- >> "$output_file"
done < "$input_file"

echo "Cleaned transcript saved to $output_file"

# Optional: Show first few lines of cleaned file
echo -e "\nFirst few lines of cleaned transcript:"
head -n 3 "$output_file"
