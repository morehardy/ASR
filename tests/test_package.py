import unittest
from unittest.mock import patch

import asr


class PackageMetadataTest(unittest.TestCase):
    def test_version_comes_from_installed_distribution_metadata(self) -> None:
        with patch("asr.metadata.version", return_value="1.2.3") as version:
            self.assertEqual(asr._read_version(), "1.2.3")
            version.assert_called_once_with("echoalign-asr-mlx")

    def test_version_has_source_checkout_fallback(self) -> None:
        with patch("asr.metadata.version", side_effect=asr.metadata.PackageNotFoundError):
            self.assertEqual(asr._read_version(), "0+unknown")

    def test_version_is_defined(self) -> None:
        self.assertTrue(asr.__version__)
