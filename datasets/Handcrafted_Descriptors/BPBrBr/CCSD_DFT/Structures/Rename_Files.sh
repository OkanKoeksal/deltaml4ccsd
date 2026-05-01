#!/bin/bash

# Get the list of structs files sorted numerically
files=(interpolated*.xyz)
IFS=$'\n' sorted_files=($(sort -V <<<"${files[*]}"))
unset IFS

# Counter to keep track of the file number
counter=1

# Loop through each sorted xyz file
for file in "${sorted_files[@]}"; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Extract the number from the filename
        num=$(echo "$file" | grep -oE '[0-9]+')
        # Rename the file to 'random' followed by the counter
        mv "$file" "interpolated$counter.xyz"
        # Increment the counter for the next file
        ((counter++))
    fi
done
