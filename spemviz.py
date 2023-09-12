"""Main function for embedding projector.

----------
Alessandro De Luca - 09/2023
"""
# TODO: GUI!

import os
import argparse
import re

from src import data_loader, embedding_projector


def main():
    parser = argparse.ArgumentParser(
        description="""SpEmViz: Speaker embedding projector using TensorBoard.
        The program takes as inputs the path to the data, the metadata variable names 
        or indices if any, the features variable names or indices or None/'all' if 
        all variables are features.
        To open the projector, click/copy and paste the first link; something like 'http://localhost:60067#projector'.
        """
    )
    parser.add_argument(
        '--path', '-d', help='Path to the data; relative to where the script is running from or absolute.',
        type=str, required=True
    )
    parser.add_argument(
        '--features', '-x', help="""Feature variables: 
            - list of strings for variable names
            - list of integers for variable indices
            - 'all' for all variables
            - None for those not in metavariables
            - tuple like (0, 5) for all variables from index 0 to index 5
            (default: None)""",
        nargs='*', default=None
    )
    parser.add_argument(
        '--metavars', '-y', help="""Metadata variables: 
            - list of strings for variable names
            - list of integers for variable indices
            - 'infer' for only variables not in features
            - None for those not in metavariables
            - tuple like (0, 5) for all variables from index 0 to index 5
            (default: None)""",
        nargs='*', default=None
    )
    parser.add_argument(
        '--log_dir', help='Directory to save logs for TensorBoard (default: ./logs/)',
        nargs='?', default='logs'
    )
    args = parser.parse_args()

    # Transform the inputs accordingly
    if type(args.features) is str:
        if args.features == 'all':
            features = args.features
        elif args.features == 'None':
            args.features = None
        elif re.search(re.escape('(', args.features)) and re.search(re.escape(')', args.features)):
            features = args.features.strip('() ')
            features = features.split(',')
            lb = int(features[0])
            ub = int(features[1])
            features = list(range(lb, ub+1))
        else:
            features = [args.features]

    if type(args.metavars) is str:
        if args.metavars == 'infer':
            metavars = args.metavars
        elif args.metavars == 'None':
            args.metavars = None
        elif re.search(re.escape('(', args.metavars)) and re.search(re.escape(')', args.metavars)):
            metavars = args.metavars.strip('() ')
            metavars = metavars.split(',')
            lb = int(metavars[0])
            ub = int(metavars[1])
            metavars = list(range(lb, ub+1))
        else:
            metavars = [args.metavars]

    ptd = os.path.join(os.getcwd(), args.path)
    log_dir = os.path.join(os.getcwd(), args.log_dir)

    # Run the data_loader function
    X, Y, metavars = data_loader.load_data(
        features=features, metavars=metavars, path_to_data=ptd
    )

    # Run embedding projector
    tbTool = embedding_projector.TensorBoardTool(
        log_dir=log_dir, embedding_vecs=X, metadata=Y, metadata_var=metavars
    )
    tbTool.run()
