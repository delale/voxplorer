import os
import unittest
import numpy as np
from ..lib.speaker_embeddings import spembed


class TestSpembed(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_dir = 'tests/data'
        self.audio_file = 'tests/data/01_test_001.wav'

    def test_spembed_dir(self):
        result = spembed(self.audio_dir)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, 192))

    def test_spembed_file(self):
        result = spembed(self.audio_file)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 192))

    def test_spembed_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            spembed('invalid_input')


if __name__ == '__main__':
    unittest.main()
