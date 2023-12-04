# FILEPATH: VoxRimor/tests/test_data_manipulator.py

import os
import unittest
from ..lib.data_manipulator import extract_metadata


class TestExtractMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.files_list = ['test/file_1_var1_var2.ext',
                           'test/file_2_var3_var4.ext']
        self.metavars = ['-', '-', 'var1', 'var2']

    def test_extract_metadata(self):
        result = extract_metadata(self.files_list, self.metavars, '_')
        self.assertIsInstance(result, dict)
        self.assertEqual(
            result,
            {
                'filename': [os.path.basename(f) for f in self.files_list],
                'var1': ['var1', 'var3'], 'var2': ['var2', 'var4']
            }
        )

    def test_extract_metadata_empty_list(self):
        with self.assertRaises(ValueError):
            extract_metadata([], self.metavars, '_')


if __name__ == '__main__':
    unittest.main()
