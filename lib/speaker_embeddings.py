# Speaker embeddings using ECAPA-TDNN VoxCeleb
# TODO: add speaker verification option

import warnings
import os
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import torch
import torch.nn.functional as F
import torchaudio


def _load_audio_files(wav_files: list) -> torch.Tensor:
    """
    Audio files loader. Also makes sure that sampling rate is 16000 Hz.

    Parameters:
    -----------
    wav_files: list
        List of audio file paths.

    Returns:
    --------
    wavs : torch.Tensor
        Tensor of audio files.
    wav_lens: torch.Tensor
        Tensor of audio file lengths relative to longest audio file.
    """
    # load files
    wavs: list = [None] * len(wav_files)
    max_len = 0
    for i, wav_file in enumerate(wav_files):
        signal, sr = torchaudio.load(wav_file)
        if sr != 16000:  # make sure sampling rate = 16000 Hz
            signal = torchaudio.functional.resample(
                signal, orig_freq=sr, new_freq=16000
            )
        if signal.shape[1] > max_len:
            max_len = signal.shape[1]

        wavs[i] = signal

    # pad files
    wav_lens = [sig.shape[1] / max_len for sig in wavs]
    wavs = [F.pad(sig, (0, max_len - sig.shape[1]),
                  mode='constant', value=0) for sig in wavs]

    # cat tensors
    wavs = torch.cat(wavs, dim=0)
    wav_lens = torch.tensor(wav_lens)

    return wavs, wav_lens


def spembed(audio_dir: str) -> np.ndarray:
    """
    Speaker embedding using ECAPA-TDNN VoxCeleb.

    Parameters:
    -----------
    audio_dir: str
        Directory containing audio files. Can also be a single audio file.

    Returns:
    --------
    np.ndarray
        Speaker embedding (N files, 192).
    """

    if os.path.isdir(audio_dir):
        audio_files: list = [
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')
        ]
        if len(audio_files) == 0:
            raise FileNotFoundError('No .wav files found in audio_dir')
    elif audio_dir.endswith('.wav'):
        audio_files: list = [audio_dir]
    else:
        raise FileNotFoundError(
            'audio_dir must be a directory containing .wav files or a .wav file')

    # load files
    wavs: torch.Tensor
    wav_lens: torch.Tensor
    wavs, wav_lens = _load_audio_files(audio_files)

    # init classifier
    if not os.path.isdir('.pretrained_classifier'):
        os.mkdir('.pretrained_classifier')
    classifier: EncoderClassifier = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='.pretrained_classifier'
    )

    # embed
    embedding: torch.Tensor = classifier.encode_batch(
        wavs=wavs, wav_lens=wav_lens)

    # reshape and transform to numpy
    embedding: torch.Tensor = torch.reshape(embedding, shape=(
        embedding.shape[0], 192)).numpy()  # 192 embeddings

    return embedding
