#!/bin/bash

# Define input and output folders
input_folder="assets/shape_variant/urdf/"
output_folder="assets/shape_variant/usd/"

# Make sure the output folder exists
mkdir -p "$output_folder"

# Iterate over all .urdf files in the input folder
for urdf_file in "$input_folder"*/model.urdf; do
    # Get the folder name (without path)
    foldername=$(basename "$(dirname "$urdf_file")")

# "$urdf_file": This is a variable containing the full path of a file, e.g., /path/to/your/file.urdf.

# dirname "$urdf_file": The dirname command returns the path excluding the filename. For example, if urdf_file is /path/to/your/file.urdf, dirname "$urdf_file" returns /path/to/your.

# basename "$(dirname "$urdf_file")": The basename command returns the last part of a path. For example, if dirname "$urdf_file" returns /path/to/your, basename "$(dirname "$urdf_file")" returns your.

    # Define the output file path
    output_file="$output_folder$foldername/model.usd"
    output_instanceable_file="$output_folder$foldername/model_instanceable.usd"

    echo $urdf_file
    echo $output_file
    ehco $output_instanceable_file

    # Make sure the output folder exists
    mkdir -p "$(dirname "$output_file")"

    # Call the convert_urdf.py script
    python source/standalone/tools/convert_urdf.py \
        "$urdf_file" \
        "$output_file" \
        --headless
    python source/standalone/tools/convert_urdf.py \
        "$urdf_file" \
        "$output_instanceable_file" \
        --make-instanceable \
        --headless
done