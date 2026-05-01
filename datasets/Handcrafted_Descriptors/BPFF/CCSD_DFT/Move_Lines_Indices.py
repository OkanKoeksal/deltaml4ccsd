def process_lines_interactive(files):
    """
    Interactively allows the user to either move or remove a range of lines across multiple files.

    :param files: List of file paths to process.
    """
    print("Welcome to the line mover/remover script!")
    print(f"The files to be processed are: {', '.join(files)}\n")

    # Step 1: Prompt the user for action
    action = input("Do you want to move or remove lines? (Enter 'move' or 'remove'): ").strip().lower()
    if action not in {"move", "remove"}:
        print("Invalid action. Please enter 'move' or 'remove'.")
        return

    # Step 2: Prompt for indices
    range_start = int(input("Enter the starting index (inclusive, starting from 1): "))
    range_end = int(input("Enter the ending index (inclusive, starting from 1): "))
    
    # Convert user-friendly 1-based indices to Python 0-based indices
    range_start -= 1
    range_end -= 1
    
    # If moving lines, prompt for target line
    target_line = None
    if action == "move":
        target_line = int(input("Enter the target line to move the selected entries to (starting from 1): "))
        target_line -= 1  # Adjust to 0-based index

    # Step 3: Process each file
    for file in files:
        print(f"\nProcessing file: {file}")
        
        # Read all lines from the file
        with open(file, 'r') as f:
            lines = f.readlines()
        
        # Validate inputs
        if range_start < 0 or range_end >= len(lines):
            print(f"Error: The range ({range_start + 1}, {range_end + 1}) is out of bounds for {file}.")
            continue
        if action == "move" and (target_line < 0 or target_line > len(lines) - (range_end - range_start + 1)):
            print(f"Error: The target line {target_line + 1} is out of bounds for {file}.")
            continue
        
        # Extract lines to move/remove
        lines_to_handle = lines[range_start:range_end + 1]
        
        # Remove these lines from the original content
        remaining_lines = lines[:range_start] + lines[range_end + 1:]
        
        if action == "move":
            # Insert the extracted lines at the target position
            updated_lines = (
                remaining_lines[:target_line]
                + lines_to_handle
                + remaining_lines[target_line:]
            )
            print(f"Lines {range_start + 1} to {range_end + 1} moved to line {target_line + 1} in {file}.")
        elif action == "remove":
            # Do not reinsert the extracted lines
            updated_lines = remaining_lines
            print(f"Lines {range_start + 1} to {range_end + 1} removed completely from {file}.")

        # Write the updated content back to the file
        with open(file, 'w') as f:
            f.writelines(updated_lines)

# List of files to process
files = [
    "CCSD_ML_Final.txt",
    "processed_xyz_files_final.txt",
    "processed_features_final.txt",
    "DFT.txt"
]

# Run the interactive function
process_lines_interactive(files)
