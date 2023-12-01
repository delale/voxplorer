import numpy as np
import unittest
import librosa
import parselmouth
from ..lib import feature_extraction as fe


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_file = 'tests/data/test.wav'
        self.y, self.sr = librosa.load(self.audio_file, sr=None)
        self.mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        self.sound = parselmouth.Sound(self.audio_file)

    def test_mel_features(self):
        result = fe.mel_features(self.audio_file)
        self.assertEqual(result.shape[1], 13)

        result = fe.mel_features(self.audio_file, deltas=True)
        self.assertEqual(result.shape[1], 13*3)

        result = fe.mel_features(self.audio_file, summarise=True)
        self.assertEqual(result.shape, (13*2,))

        result = fe.mel_features(self.audio_file, summarise=True, deltas=True)
        self.assertEqual(result.shape, (13*6,))

    def test_delta_delta(self):
        result = fe._delta_delta(self.mfccs)
        self.assertEqual(result.shape[0], 13*3)

    def test_summarise_mfccs(self):
        result = fe._summarise_features(self.mfccs)
        self.assertEqual(result.shape[0], 13*2)

    def test_picth(self):
        pitch_features = fe._pitch(self.sound)

        # Tuple of length 5?
        self.assertIsInstance(pitch_features, tuple)
        self.assertEqual(len(pitch_features), 5)

        # Tuple of floats?
        for feature in pitch_features:
            self.assertIsInstance(feature, float)

    def test_formants(self):
        formant_features = fe._formants(self.sound)

        # Tuple of length 4
        self.assertIsInstance(formant_features, tuple)
        self.assertEqual(len(formant_features), 4)

        # Tuple of floats?
        for feature in formant_features:
            self.assertIsInstance(feature, float)

    def test_vocal_tract_estimates(self):
        formants = fe._formants(self.sound)
        vte_features = fe._vocal_tract_estimates(self.sound, formants)

        # Tuple of length 5
        self.assertIsInstance(vte_features, tuple)
        self.assertEqual(len(vte_features), 5)

        # Tuple of floats?
        for feature in vte_features:
            self.assertIsInstance(feature, float)

    def test_hnr(self):
        hnr = fe._hnr(self.sound)

        # Value is float?
        self.assertIsInstance(hnr, float)

    def test_jitter_shimmer(self):
        jitter, shimmer = fe._jitter_shimmer(self.sound)

        # Values are floats?
        self.assertIsInstance(jitter, float)
        self.assertIsInstance(shimmer, float)

    def test_acoustic_features(self):
        result = fe.acoustic_features(self.audio_file)

        # Result is numpy array with shape (18,)?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (18,))

        # Array of floats?
        for feature in result:
            self.assertIsInstance(feature, float)

    def test_low_lvl_features(self):
        result = fe.low_lvl_features(self.audio_file)

        # Result is numpy array of 12 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 12)

        # Test with use_mean_contrasts=True
        result = fe.low_lvl_features(self.audio_file, use_mean_contrasts=True)

        # Result is numpy array of 6 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 6)

        # Test with summarise = True
        result = fe.low_lvl_features(self.audio_file, summarise=True)

        # Result is numpy array of 24 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (24,))

        # Test with summarise = True and use_mean_contrasts=True
        result = fe.low_lvl_features(
            self.audio_file, summarise=True, use_mean_contrasts=True)

        # Result is numpy array of 12 features?
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (12,))

    def test_lpcc(self):
        # Test with default params
        result = fe.lpcc(self.audio_file)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 13)

        # Test with different n_lpcc
        result = fe.lpcc(self.audio_file, n_lpcc=20)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 20)

        # Test with summarise=True
        result = fe.lpcc(self.audio_file, summarise=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (13*2,))

        # Test with summarise=True and different n_lpcc
        result = fe.lpcc(self.audio_file, summarise=True, n_lpcc=20)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (20*2,))


if __name__ == '__main__':
    unittest.main()
