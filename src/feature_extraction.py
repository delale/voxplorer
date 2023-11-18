# TODO: create mfccs extraction with possibility to have:
#   - deltas (both or only first)
#   - summarised by utterance

import numpy as np
import librosa

# TODO: create function to extract mfccs (and delta-delta) from audio file


# helper of mel_features() to extract delta and delta-delta coefficients
def _delta_delta(mfccs: np.ndarray):
    """
    Computes delta and delta-delta coefficients from MFCCs.

    Parameters:
    -----------
    mfccs : np.ndarray
        MFCCs for each frame in the audio file. Shape is (n_frames, n_mfccs).

    Returns:
    --------
    features_vec : np.ndarray
        updated features vector containing MFCCs, deltas, and delta-deltas
        for each frame in the audio file. Shape is (n_frames, n_mfccs*3).
    """
    # compute delta and delta-delta
    delta = librosa.feature.delta(mfccs, order=1)
    delta_delta = librosa.feature.delta(mfccs, order=2)

    # concatenate features
    features_vec = np.concatenate((mfccs, delta, delta_delta), axis=1)

    return features_vec


# helper of mel_features() to summarise mfccs by utterance
def _summarise_mfccs(mfccs: np.ndarray):
    pass


# MFCCs extraction
def mel_features(
    audio_file: str,
    n_mfcc: int = 13,
    n_mels: int = 40,
    win_length: float = 25,
    overlap: float = 10,
    fmin: int = 150,
    fmax: int = 4000,
    premphasis: float = 0.95,
    lifter: int = 22,
    deltas: bool = False,
    summarise: bool = False
):
    """
    Extracts mfccs (and optionally delta-delta) from audio file.

    Parameters:
    -----------
    audio_file : str
        Path to audio file.
    n_mfcc : int
        Number of MFCCs to return.
    n_mels : int
        Number of mel bands to generate.
    win_length : float
        Window length in milliseconds.
    overlap : float
        Overlap length in milliseconds.
    fmin : int
        Minimum frequency in mel filterbank in Hz.
    fmax : int
        Maximum frequency in mel filterbank in Hz.
    premphasis : float
        Coefficient for pre-emphasis filter.
    lifter : int
        Liftering coefficient.
    deltas : bool
        Whether to return delta and delta-delta coefficients.
    summarise : bool
        Whether to summarise MFCCs by utterance. If True, returns mean and standard
        deviation of MFCCs for each utterance.

    Returns:
    --------
    features_vec : np.ndarray
        MFCCs for each frame in the audio file. If deltas is True, contains also 
        delta and delta-delta coefficients. If summarise is True, contains mean and
        standard deviation of MFCCs for each utterance. Shape is (n_frames, n_mfccs).
    """
    y, sr = librosa.load(audio_file, sr=None)  # load audio file

    # pre-emphasis filter
    y = librosa.effects.preemphasis(y, coef=premphasis)

    # compute frame length and overlap in samples
    n_fft = int(win_length * sr / 1000)
    hop_length = int(overlap * sr / 1000)

    # extract mfccs
    features_vec = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hamming',
        lifter=lifter,
        fmin=fmin,
        fmax=fmax
    )

    # delta features
    if deltas:
        features_vec = _delta_delta(features_vec)

    # summarise by utterance
    if summarise:
        features_vec = _summarise_mfccs(features_vec)

    return features_vec.T


# TODO: create function to extract spectral features
# picth, formants, hnr, jittter, shimmer, spectral entropy,
# energy, zero-crossing-rate, spectral bandwidth, spectral contrast,
# spectral roll-off, formant dispersion

# TODO: create function to extract LPCCs and LPC residual
