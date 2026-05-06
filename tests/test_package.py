import unittest
import tomllib
from pathlib import Path
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

    def test_mlx_extra_includes_torchcodec_for_silero_vad_audio_io(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        mlx_dependencies = pyproject["project"]["optional-dependencies"]["mlx"]

        self.assertIn("torchcodec>=0.11.1", mlx_dependencies)
