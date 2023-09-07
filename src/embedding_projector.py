"""Tensorflow embedding projector to visualize different speaker embeeddings.

----------
Alessandro De Luca - 09/2023
"""
import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


def _data_setup(
        embedding_vecs: np.ndarray, metadata: np.ndarray,
        metadata_var: list, log_dir='for_tensorboard/logs/') -> tf.Variable:
    """Prepares data for embedding projector.

    Args:
        embedding_vecs: Array of embedding vectors.
        metadata: Array of metadata variables.
        metadata_vars: Array of metadata variable names 

    Returns:
        embedding_var [tf.Variable]
    """

    # Make log directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Embedding variable initialized with embeddings
    embedding_var = tf.Variable(embedding_vecs, name='embedding')

    # Make metadata variable name a list if not already
    if type(metadata_var) is not list:
        metadata_var = [metadata_var]

    # Reshape metadata if 1D
    if len(metadata_var) == 1 and len(metadata.shape) == 1:
        metadata = metadata.reshape((metadata.shape[0], 1))

    # Metadata file
    with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        # Write labels for metadata
        f.write('\t'.join(metadata_var)+"\n")

        # Write metadata labels
        for labels in metadata:
            f.write('\t'.join(labels)+"\n")

    return embedding_var


def projector_setup(
        embedding_vecs: np.ndarray, metadata: np.ndarray,
        metadata_var: list, log_dir='for_tensorboard/logs/') -> None:
    """Sets up embedding projector and projects the embeddings.

    Args
        embedding_vecs: Array of embedding vectors.
        metadata: Array of metadata variables.
        metadata_vars: Array of metadata variable names 
    """
    # Prepare data for embedding projector
    embedding_var = _data_setup(
        embedding_vecs, metadata, metadata_var, log_dir)

    # Configure embedding projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embedding'
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)


def test():
    import pickle

    # Load data
    with open('testing/test_mfccs.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    with open('testing/test_sp.pkl', 'rb') as f:
        metadata = pickle.load(f)

    metadata_vars = ['speaker']

    # Run tensorflow embedding projector setup
    projector_setup(embedding_vecs=embeddings, metadata=metadata,
                    metadata_var=metadata_vars, log_dir='for_tensorboard/logs/test/')

    # Launch TensorBoard
    subprocess.run('tensorboard --logdir for_tensorboard/logs/test/')


if __name__ == '__main__':
    test()
