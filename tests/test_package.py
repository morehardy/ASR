import unittest
import tomllib
from pathlib import Path
from unittest.mock import patch

import asr

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_pyproject() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


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
        pyproject = read_pyproject()
        mlx_dependencies = pyproject["project"]["optional-dependencies"]["mlx"]

        self.assertIn("torchcodec>=0.11.1", mlx_dependencies)

    def test_package_metadata_includes_discovery_fields(self) -> None:
        pyproject = read_pyproject()
        project = pyproject["project"]

        self.assertIn("speech-recognition", project["keywords"])
        self.assertIn("subtitles", project["keywords"])
        self.assertEqual(project["license"], "MIT")
        self.assertIn("Operating System :: MacOS", project["classifiers"])
        self.assertIn("License :: OSI Approved :: MIT License", project["classifiers"])
        self.assertIn("Topic :: Multimedia :: Sound/Audio :: Speech", project["classifiers"])
        self.assertEqual(
            project["urls"]["Repository"],
            "https://github.com/morehardy/echoalign-asr-mlx",
        )
        self.assertEqual(
            project["urls"]["Issues"],
            "https://github.com/morehardy/echoalign-asr-mlx/issues",
        )
