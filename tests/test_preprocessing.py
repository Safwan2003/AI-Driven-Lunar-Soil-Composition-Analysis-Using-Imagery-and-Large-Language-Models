# tests/test_preprocessing.py

"""
Unit tests for the data preprocessing scripts.
"""

import unittest
import numpy as np
import sys
sys.path.append('../src')

from data.preprocessing import denoise_image, normalize_image

class TestPreprocessing(unittest.TestCase):

    def test_denoise_image(self):
        """Tests the denoising function."""
        # Create a dummy image
        image = np.random.rand(100, 100, 3)
        denoised = denoise_image(image)
        self.assertEqual(image.shape, denoised.shape)

    def test_normalize_image(self):
        """Tests the normalization function."""
        # Create a dummy image
        image = np.random.randint(0, 256, size=(100, 100, 3)).astype(float)
        normalized = normalize_image(image)
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))

if __name__ == '__main__':
    unittest.main()
