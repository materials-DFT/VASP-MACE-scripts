#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def reorder_poscar(poscar_data, new_order):
    """
    Reorders the atomic coordinates in a VASP POSCAR file based on a new element order.

    Args:
        poscar_data (str): The content of the POSCAR file as a string.
        new_order (list): A list of strings specifying the desired order of elements.

    Returns:
        str: The new POSCAR content with reordered coordinates and element information.
    """
    # Split the input data into lines and clean up whitespace
    lines = [line.strip() for line in poscar_data.strip().split('\n') if line.strip()]

    # Check for empty or short POSCAR
    if len(lines) < 8:
        raise ValueError("POSCAR file is too short to be valid.")

    # The header includes the title, scaling factor, lattice vectors, element names,
    # counts, and the coordinate system type (Cartesian or Direct).
    # Lines 0-4 are constant, lines 5-6 contain element info, line 7 is the coordinate type.
    header_lines = lines[:8]

    # Extract element types and counts from the header
    old_elements = header_lines[5].split()
    old_counts = [int(c) for c in header_lines[6].split()]

    # Validate that all elements in the new order are present in the original file
    if not all(elem in old_elements for elem in new_order):
        raise ValueError("New order contains elements not present in the original POSCAR.")

    # The atomic coordinates start from line 9 (index 8)
    coords_lines = lines[8:]

    # Group coordinates by their element type based on the original POSCAR structure
    coord_dict = {elem: [] for elem in old_elements}
    current_index = 0
    for i, elem in enumerate(old_elements):
        count = old_counts[i]
        # Collect the coordinate lines for the current element
        coord_dict[elem] = coords_lines[current_index:current_index + count]
        current_index += count

    # Create the new element and count lists, and reordered coordinates
    new_elements = []
    new_counts = []
    new_coords = []

    for elem in new_order:
        if elem in coord_dict:
            new_elements.append(elem)
            count = len(coord_dict[elem])
            new_counts.append(count)
            # Add the coordinates for this element to the new list
            new_coords.extend(coord_dict[elem])

    # Construct the final reordered POSCAR file content
    new_poscar = (
        header_lines[0] + '\n' +
        header_lines[1] + '\n' +
        '\n'.join(header_lines[2:5]) + '\n' +
        '  ' + ' '.join(new_elements) + '\n' +
        '  ' + ' '.join(str(c) for c in new_counts) + '\n' +
        header_lines[7] + '\n' +
        '\n'.join(new_coords)
    )

    return new_poscar

def main():
    """
    Main function to process POSCAR files from command-line input.
    """
    if len(sys.argv) < 2:
        print("Usage: python reorder_poscar.py <file_or_directory_path>")
        sys.exit(1)

    path = sys.argv[1]
    new_order = ['K', 'Mn', 'O']

    if os.path.isfile(path):
        process_file(path, new_order)
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            if 'POSCAR' in filenames:
                file_path = os.path.join(dirpath, 'POSCAR')
                process_file(file_path, new_order)
    else:
        print(f"Error: The path '{path}' does not exist or is not a valid file/directory.", file=sys.stderr)
        sys.exit(1)

def process_file(file_path, new_order):
    """
    Reads, reorders, and overwrites a single POSCAR file.
    """
    print(f"Processing {file_path}...")
    try:
        with open(file_path, 'r') as f:
            poscar_content = f.read()

        reordered_content = reorder_poscar(poscar_content, new_order)

        with open(file_path, 'w') as f:
            f.write(reordered_content)

        print(f"Successfully reordered and overwrote {file_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except ValueError as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred with {file_path}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()

