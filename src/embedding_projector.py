"""Tensorflow embedding projector to visualize different speaker embeeddings.

----------
Alessandro De Luca - 09/2023
"""
import os
import subprocess
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard import program
from tensorboard.plugins import projector


class TensorBoardTool:

    def __init__(self, log_dir, embedding_vecs, metadata, metadata_var):
        """Initialize TensorBoardTool class.

        Args:
            log_dir: Path to log directory for TensorBoard.
            embedding_vecs: Array of embedding vectors.
            metadata: Array of metadata variables.
            metadata_vars: Array of metadata variable names.
        """
        self.log_dir = log_dir
        self.embedding_vecs = embedding_vecs

        # Make metadata variable name a list if not already
        if type(metadata_var) is not list:
            self.metadata_var = [metadata_var]
        else:
            self.metadata_var = metadata_var

        # Reshape metadata if 1D
        if len(self.metadata_var) == 1 and len(metadata.shape) == 1:
            self.metadata = metadata.reshape((metadata.shape[0], 1))
        else:
            self.metadata = metadata

    def _data_setup(self) -> None:
        """Prepares data for embedding projector.
        """
        # Make log directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create embedding tsv file
        pd.DataFrame(
            data=self.embedding_vecs
        ).to_csv(os.path.join(self.log_dir, 'embeddings.tsv'),
                 sep='\t', header=False, index=False)

        # Metadata file
        with open(os.path.join(self.log_dir, 'metadata.tsv'), 'w') as f:
            if len(self.metadata_var) > 1:
                # Write labels for metadata if there are more than 1 vars
                f.write('\t'.join(self.metadata_var)+"\n")

            # Write metadata labels
            for labels in self.metadata:
                f.write('\t'.join(labels)+"\n")

    def _projector_setup(self) -> None:
        """Sets up embedding projector and projects the embeddings.
        """
        # Prepare data for embedding projector
        self._data_setup()

        # Configure embedding projector
        self.config = projector.ProjectorConfig()
        self.embedding = self.config.embeddings.add()
        self.embedding.tensor_path = 'embeddings.tsv'
        self.embedding.metadata_path = 'metadata.tsv'
        projector.visualize_embeddings(self.log_dir, self.config)

    def run(self):
        self._projector_setup()
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        tb.main()


def test():
    import pickle
    log_dir = 'for_tensorboard/logs/test/'

    # Load data
    with open('testing/test_mfccs.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open('testing/test_sp.pkl', 'rb') as f:
        metadata = pickle.load(f)

    metadata_vars = ['speaker']

    # Run tensorflow embedding projector setup
    tbTool = TensorBoardTool(
        log_dir=log_dir, embedding_vecs=embeddings,
        metadata=metadata, metadata_var=metadata_vars
    )
    tbTool.run()


if __name__ == '__main__':
    test()
