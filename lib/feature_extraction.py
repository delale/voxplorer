# TODO: create mfccs extraction with possibility to have:
#   - deltas (both or only first)
#   - summarised by utterance
from typing import Tuple
import numpy as np
import librosa
import parselmouth


# helper of mel_features() to extract delta and delta-delta coefficients
def _delta_delta(mfccs: np.ndarray) -> np.ndarray:
    """
    Computes delta and delta-delta coefficients from MFCCs.

    Parameters:
    -----------
    mfccs : np.ndarray
        MFCCs for each frame in the audio file.

    Returns:
    --------
    features_vec : np.ndarray
        updated features vector containing MFCCs, deltas, and delta-deltas
        for each frame in the audio file. Shape is (n_frames, n_mfccs*3).
    """
    # compute delta and delta-delta
    delta: np.ndarray = librosa.feature.delta(mfccs, order=1)
    delta_delta: np.ndarray = librosa.feature.delta(mfccs, order=2)

    # concatenate features
    features_vec: np.ndarray = np.concatenate(
        (mfccs, delta, delta_delta), axis=0)

    return features_vec


# helper to summarise features by utterance
def _summarise_features(features_vec: np.ndarray) -> np.ndarray:
    """
    Summarises features by utterance.

    Parameters:
    -----------
    features_vec : np.ndarray
        features vector for each frame in the audio file. Shape is (n_features, n_frames).

    Returns:
    --------
    np.ndarray
        Mean and standard deviation of features for each utterance. Shape is
        (n_features*2,)
    """
    return np.concatenate(
        (np.mean(features_vec, axis=1), np.std(features_vec, axis=1)), axis=0
    )


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
) -> np.ndarray:
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
    y: np.ndarray
    sr: int
    y, sr = librosa.load(audio_file, sr=None)  # load audio file

    # pre-emphasis filter
    y: np.ndarray = librosa.effects.preemphasis(y, coef=premphasis)

    # compute frame length and overlap in samples
    n_fft: int = int(win_length * sr / 1000)
    hop_length: int = int(overlap * sr / 1000)

    # extract mfccs
    features_vec: np.ndarray = librosa.feature.mfcc(
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
        features_vec: np.ndarray = _delta_delta(features_vec)

    # summarise by utterance
    if summarise:
        features_vec: np.ndarray = _summarise_features(features_vec)

    return features_vec.T


# pitch helper
def _pitch(
    sound: parselmouth.Sound, time_step: float = 0.0, f0min: float = 75.0, f0max: float = 600.0,
    max_candidates: int = 15, silence_threshold: float = 0.03,
    voicing_threshold: float = 0.45, octave_cost: float = 0.01,
    octave_jump_cost: float = 0.35, voiced_unvoiced_cost: float = 0.14
) -> Tuple[float, ...]:
    """
    Extracts pitch features from audio file using cross-correlation method.

    Parameters:
    -----------
    sound : parselmouth.Sound
        Parselmouth sound object.
    time_step : float
        Time step in seconds (default=0.0 (=auto)).
    f0min : float
        Minimum pitch frequency in Hz.
    f0max : float
        Maximum pitch frequency in Hz.
    max_candidates : int
        Maximum number of candidates.
    silence_threshold : float
        Threshold for silence.
    voicing_threshold : float
        Threshold for voicing.
    octave_cost : float
        Cost for octave.
    octave_jump_cost : float
        Cost for octave jump.
    voiced_unvoiced_cost : float
        Cost for unvoiced.

    Returns:
    --------
    pitch_features : Tuple[float, ...]
        Pitch features for each frame in the audio file. Shape is (n_frames,).
        The extracted features are:
        - mean pitch
        - median pitch
        - minimum pitch
        - maximum pitch
        - standard deviation of pitch
    """
    pitch: parselmouth.Pitch = parselmouth.praat.call(
        sound, "To Pitch (cc)",
        time_step, f0min, max_candidates, True, silence_threshold,
        voicing_threshold, octave_cost, octave_jump_cost,
        voiced_unvoiced_cost, f0max
    )
    mean_pitch: float = parselmouth.praat.call(
        pitch, 'Get mean', 0, 0, 'Hertz')
    med_pitch: float = parselmouth.praat.call(
        pitch, 'Get quantile', 0, 0, 0.5, 'Hertz')
    min_pitch: float = parselmouth.praat.call(
        pitch, 'Get minimum', 0, 0, 'Hertz', 'Parabolic')
    max_pitch: float = parselmouth.praat.call(
        pitch, 'Get maximum', 0, 0, 'Hertz', 'Parabolic')
    std_pitch: float = parselmouth.praat.call(
        pitch, 'Get standard deviation', 0, 0, 'Hertz')
    return mean_pitch, med_pitch, min_pitch, max_pitch, std_pitch


# formants helper
def _formants(
    sound: parselmouth.Sound, maximum_formant: int = 5000
) -> Tuple[float, ...]:
    """
    Extracts formants from audio file using Burg method.

    Parameters:
    -----------
    sound : parselmouth.Sound
        Parselmouth sound object.
    maximum_formant : int
        Maximum formant frequency in Hz.

    Returns:
    --------
    formants_features : Tuple[float, ...]
        Mean F1, F2, F3, F4.
    """
    formants: parselmouth.Formant = sound.to_formant_burg(
        maximum_formant=maximum_formant)
    f1: float = parselmouth.praat.call(formants, 'Get mean', 1, 0, 0, 'Hertz')
    f2: float = parselmouth.praat.call(formants, 'Get mean', 2, 0, 0, 'Hertz')
    f3: float = parselmouth.praat.call(formants, 'Get mean', 3, 0, 0, 'Hertz')
    f4: float = parselmouth.praat.call(formants, 'Get mean', 4, 0, 0, 'Hertz')

    return f1, f2, f3, f4


# Helper for vocal tract estimates; code inspired from Voicelab (github.com/Voice-Lab)
def _vocal_tract_estimates(
    sound: parselmouth.Sound, formants: Tuple[float, ...],
) -> Tuple[float, ...]:
    """
    Extract vocal tract estimates measures from audio file.

    Parameters:
    -----------
    sound : parselmouth.Sound
        Parselmouth sound object.
    formants : Tuple[float, ...]
        Mean formants in the audio file (F1, F2, F3, F4).

    Returns:
    --------
    formant_dispersion : float
        Formant dispersion.
    avg_formant : float
        Average formant.
    geometric_mean : float
        Geometric mean.
    fitch_vtl : float
        Fitch VTL.
    delta_f : float
        Delta f.
    """
    # Extract formant means
    f1: float = formants[0]
    f2: float = formants[1]
    f3: float = formants[2]
    f4: float = formants[3]

    # Formant dispersion
    formant_dispersion: float = (f4 - f1) / 3

    # Average formant
    avg_formant: float = (f1 + f2 + f3 + f4) / 4

    # Geometric mean
    geometric_mean: float = (f1 * f2 * f3 * f4) ** 0.25

    # Fitch VTL
    fitch_vtl: float = ((1 * (35000 / (4 * f1))) + (3 * (35000 / (4 * f2))) +
                        (5 * (35000 / (4 * f3))) + (7 * (35000 / (4 * f4)))) / 4

    # delta f
    xysum: float = (0.5 * f1) + (1.5 * f2) + (2.5 * f3) + (3.5 * f4)
    xsquaredsum: float = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    delta_f: float = xysum / xsquaredsum

    return formant_dispersion, avg_formant, geometric_mean, fitch_vtl, delta_f


# Helper for harmonics-to-noise ratio
def _hnr(
    sound: parselmouth.Sound, time_step: float = 0.01, fmin: float = 75.0,
    silence_threshold: float = 0.1, periods_per_window: float = 1.0
) -> float:
    """
    Extracts harmonics-to-noise ratio from audio file using cc method.

    Parameters:
    -----------
    sound : parselmouth.Sound
        Parselmouth sound object.
    time_step : float
        Time step in seconds (default=0.01).
    fmin : float
        Minimum pitch frequency in Hz.
    silence_threshold : float
        Threshold for silence.
    periods_per_window : float
        Number of periods per window.

    Returns:
    --------
    hnr : float
        Harmonics-to-noise ratio.
    """
    harmonicity: parselmouth.Harmonicity = parselmouth.praat.call(
        sound, "To Harmonicity (cc)",
        time_step, fmin, silence_threshold, periods_per_window
    )
    hnr: float = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
    return hnr


# Helper for jitter and shimmer
def _jitter_shimmer(
    sound: parselmouth.Sound, fmin: float = 75.0, fmax: float = 600.0
) -> Tuple[float, float]:
    """
    Extracts jitter and shimmer from audio file using cc method.

    Parameters:
    -----------
    sound : parselmouth.Sound
        Parselmouth sound object.
    fmin : float
        Minimum pitch frequency in Hz.
    fmax : float
        Maximum pitch frequency in Hz.

    Returns:
    --------
    jitter : float
        Jitter.
    shimmer : float
        Shimmer.
    """
    point_process: parselmouth.Data = parselmouth.praat.call(
        sound, "To PointProcess (periodic, cc)",
        fmin, fmax
    )

    jitter: float = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    shimmer: float = parselmouth.praat.call(
        [sound,
            point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    return jitter, shimmer


# Helper for rms_energy
def _rms_energy(sound: parselmouth.Sound) -> float:
    """
    Extracts rms energy from audio file.

    Parameters:
    -----------
    sound : parselmouth.Sound
        Parselmouth sound object.

    Returns:
    --------
    rms_energy : float
        Root mean square energy.
    """
    amplitude_square: np.ndarray = sound.values ** 2
    mean_amplitude_square: float = np.mean(amplitude_square)
    rms_energy: float = np.sqrt(mean_amplitude_square)
    return rms_energy


# Create function to extract common acoustic features
def acoustic_features(
        audio_file: str,
        f0min: float = 75.0, f0max: float = 600.0,
) -> np.ndarray:
    """
    Extracts acoustic features from audio file.

    Parameters:
    -----------
    audio_file : str
        Path to audio file.
    f0min : float
        Minimum pitch frequency in Hz.
    f0max : float
        Maximum pitch frequency in Hz.

    Returns:
    --------
    features_vec : np.ndarray
        Acoustic features for each frame in the audio file. Shape is (n_features,).
        The extracted features are:
        - pitch (cross-correlation): mean, median, minimum, maximum, standard deviation
        - formants (burg): F1, F2, F3, F4 averages
        - formant dispersion
        - average formant
        - geometric mean of formants
        - fitch vtl
        - delta f
        - hnr
        - jitter
        - shimmer
        - energy (rms)
    """
    # Load audio
    sound: parselmouth.Sound = parselmouth.Sound(audio_file)

    # Pitch
    pitch_features: Tuple[float, ...] = _pitch(
        sound=sound, f0min=f0min, f0max=f0max
    )

    # Formants
    maximum_formant: int
    if pitch_features[0] < 120:
        maximum_formant = 5000
    elif pitch_features[0] >= 120:
        maximum_formant = 5500
    else:
        maximum_formant = 5500
    formants_features: Tuple[float, ...] = _formants(
        sound=sound, maximum_formant=maximum_formant)

    # Vocal tract estimates
    vt_estimates = _vocal_tract_estimates(
        sound=sound, formants=formants_features)

    # HNR
    hnr: float = _hnr(sound=sound, fmin=f0min)

    # Jitter & Shimmer
    jitter: float
    shimmer: float
    jitter, shimmer = _jitter_shimmer(sound=sound)

    # RMS energy
    rms_energy: float = _rms_energy(sound=sound)

    # Concatenate features
    features_vec: np.ndarray = np.concatenate(
        (pitch_features, formants_features, vt_estimates,
         (hnr, jitter, shimmer, rms_energy)), axis=0
    )
    return features_vec


# Low level features
def low_lvl_features(
    audio_file: str, win_length: float = 25, overlap: float = 10,
    premphasis: float = 0.95, use_mean_contrasts: bool = False,
    summarise: bool = False
) -> np.ndarray:
    """
    Extract low-level features from audio file.

    Parameters:
    -----------
    audio_file : str
        Path to audio file.
    win_length : float
        Window length in milliseconds.
    overlap : float
        Overlap length in milliseconds.
    premphasis : float
        Coefficient for pre-emphasis filter.
    use_mean_contrasts : bool
        Take the mean of spectral contrasts over spectral bands. default=False.
    summarise : bool
        Whether to summarise features by utterance. If True, returns mean and standard
        deviation of features for each utterance.

    Returns:
    --------
    features_vec : np.ndarray
        Low-level features for each frame in the audio file. Shape is (n_frames, n_features).
        The extracted features are:
        - spectral centroid
        - spectral bandwidth
        - spectral contrasts
        - spectral flatness
        - spectral rolloff
        - zero crossing rate
    """
    # Load audio
    y: np.ndarray
    sr: int
    y, sr = librosa.load(audio_file, sr=None)  # load audio file

    # pre-emphasis filter
    y: np.ndarray = librosa.effects.preemphasis(y, coef=premphasis)

    # compute frame length and overlap in samples
    n_fft: int = int(win_length * sr / 1000)
    hop_length: int = int(overlap * sr / 1000)

    # Spectral centroid
    spectral_centroids: np.ndarray = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
    )

    # Spectral bandwidth
    spectral_bandwidths: np.ndarray = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
    )

    # Spectral contrast
    spectral_contrasts: np.ndarray = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
    )
    if use_mean_contrasts:
        spectral_contrasts: np.ndarray = np.mean(
            spectral_contrasts, axis=0, keepdims=True
        )

    # Spectral flatness
    spectral_flatness: np.ndarray = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length, window='hamming'
    )

    # Spectral rolloff
    spectral_rolloff: np.ndarray = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, window='hamming'
    )

    # Zero crossing rate
    zero_crossing_rate: np.ndarray = librosa.feature.zero_crossing_rate(
        y=y, frame_length=n_fft, hop_length=hop_length
    )

    # Concatenate features
    features_vec: np.ndarray = np.concatenate(
        (spectral_centroids, spectral_bandwidths, spectral_contrasts,
         spectral_flatness, spectral_rolloff, zero_crossing_rate), axis=0
    )

    # Summarise by utterance
    if summarise:
        features_vec: np.ndarray = _summarise_features(features_vec)

    return features_vec.T


# TODO: create function to extract LPCCs and LPC residual
