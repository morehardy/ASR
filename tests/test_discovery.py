import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr.discovery import discover_media_files
from asr.output import build_output_path


class DiscoveryTest(unittest.TestCase):
    def test_directory_scan_is_non_recursive_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_text("x", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.wav").write_text("x", encoding="utf-8")

            files = discover_media_files(root, recursive=False)

            self.assertEqual(files, [root / "a.mp4"])

    def test_output_path_keeps_relative_structure(self) -> None:
        source = Path("/project/media/nested/demo.wav")
        root = Path("/project/media")

        target = build_output_path(
            source=source,
            input_root=root,
            output_root=root / "outputs",
            suffix=".srt",
        )

        self.assertEqual(target, Path("/project/media/outputs/nested/demo.srt"))
