"""
This module contains functions for manipulating data.
If feature extraction mode was selected, the data is split into features and metadata variables;
It is then saved in the selected output directory and sent to the embedding projector from the main script.

If visualizer mode was selected, then the function is used to create a selection partition if the user has selected
to do so.
"""

import os
import pandas as pd


def filter_selection(output_file: str) -> None:
    """
    Filters the selection partition from the data and saves it in the same folder 
    with the addition of '_selection' to the filename.

    Parameters:
    ----------- 
    output_file: str
        Path to the output file.
    """

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        raise FileNotFoundError("File not found.")

    # Filter the selection partition
    df = df[df['selection'] == 1]

    # Save the selection partition
    output_dir = os.path.dirname(output_file)
    output_basename = os.path.splitext(os.path.basename(output_file))[0]
    output_filename = os.path.join(
        output_dir, output_basename + '_selection.csv'
    )
    df.to_csv(output_filename, index=False)


def extract_metadata(filename: str, metavars: list, separator: str = '_', ) -> dict:
    """
    Extracts metadata variables from the filename.

    Parameters:
    -----------
    filename: str
        Filename.
    metavars: list
        List of metadata variable names. If '-' then it is skipped.
    separator: str
        Separator character between metadata variables.

    Returns:
    --------
    dict
        Dictionary of metadata variables.
    """
    # Create empty dictionary
    metadata_dict = {}

    # Extract metadata from filename
    basename = os.path.basename(filename)
    metadata = os.path.splitext(basename)[0]
    metadata = metadata.split(separator)

    # append to dictionary a dictionary of metadata
    metadata_dict['filename'] = [basename]
    for j, var in enumerate(metavars):
        if var == '-':
            continue
        metadata_dict[var] = [metadata[j]]

    return metadata_dict
