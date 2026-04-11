import io
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from asr.cli import app, build_parser, main, resolve_cli_inputs, run_environment_preflight


class CliTyperBootstrapTest(unittest.TestCase):
    def test_app_symbol_is_available(self) -> None:
        self.assertTrue(callable(app))

    @patch("asr.cli.discover_cli_sources")
    @patch("asr.cli.run_environment_preflight")
    def test_main_exits_early_with_readable_error_when_preflight_fails(
        self,
        mock_preflight,
        mock_discover,
    ) -> None:
        mock_discover.return_value = [(Path("demo.mov"), Path.cwd())]
        mock_preflight.return_value = (False, "MLX unavailable")

        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            exit_code = main(["demo.mov"])

        self.assertEqual(exit_code, 1)
        self.assertIn("environment check failed", stderr.getvalue())
        self.assertIn("MLX unavailable", stderr.getvalue())


class CliParserTest(unittest.TestCase):
    def test_defaults_to_current_directory_when_input_missing(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(resolve_cli_inputs(args.inputs), [Path.cwd()])

    def test_recursive_is_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["./media", "--recursive"])

        self.assertTrue(args.recursive)


class CliEnvironmentPreflightTest(unittest.TestCase):
    @patch("asr.cli.subprocess.run")
    @patch("asr.cli.shutil.which")
    def test_preflight_succeeds_when_dependencies_are_available(self, mock_which, mock_run) -> None:
        mock_which.side_effect = ["/usr/bin/ffmpeg", "/usr/bin/ffprobe"]
        mock_run.return_value = subprocess.CompletedProcess(
            args=["python", "-c", "import mlx.core"],
            returncode=0,
            stdout="",
            stderr="",
        )

        ok, message = run_environment_preflight()

        self.assertTrue(ok)
        self.assertEqual(message, "")

    @patch("asr.cli.shutil.which")
    def test_preflight_fails_when_ffmpeg_or_ffprobe_missing(self, mock_which) -> None:
        mock_which.side_effect = [None, "/usr/bin/ffprobe"]

        ok, message = run_environment_preflight()

        self.assertFalse(ok)
        self.assertIn("ffmpeg", message)

    @patch("asr.cli.subprocess.run")
    @patch("asr.cli.shutil.which")
    def test_preflight_surfaces_mlx_failure_message(self, mock_which, mock_run) -> None:
        mock_which.side_effect = ["/usr/bin/ffmpeg", "/usr/bin/ffprobe"]
        mock_run.return_value = subprocess.CompletedProcess(
            args=["python", "-c", "import mlx.core"],
            returncode=134,
            stdout="",
            stderr="libmlx init failed",
        )

        ok, message = run_environment_preflight()

        self.assertFalse(ok)
        self.assertIn("MLX/Metal preflight failed", message)
        self.assertIn("libmlx init failed", message)

    @patch("asr.cli.discover_cli_sources")
    @patch("asr.cli.run_environment_preflight")
    def test_main_exits_early_with_readable_error_when_preflight_fails(
        self,
        mock_preflight,
        mock_discover,
    ) -> None:
        mock_discover.return_value = [(Path("demo.mov"), Path.cwd())]
        mock_preflight.return_value = (False, "MLX unavailable")

        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            exit_code = main(["demo.mov"])

        self.assertEqual(exit_code, 1)
        self.assertIn("environment check failed", stderr.getvalue())
        self.assertIn("MLX unavailable", stderr.getvalue())


class CliCompletionOutputTest(unittest.TestCase):
    @patch("asr.cli.build_fish_completion_script")
    def test_completion_fish_prints_script(self, mock_build_script) -> None:
        mock_build_script.return_value = "complete -c asr -f\n"

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = main(["completion", "fish"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "complete -c asr -f\n")

    @patch("asr.cli.build_fish_completion_script")
    def test_completion_fish_returns_error_when_generation_fails(self, mock_build_script) -> None:
        mock_build_script.side_effect = RuntimeError("generation failed")

        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            exit_code = main(["completion", "fish"])

        self.assertEqual(exit_code, 1)
        self.assertIn("completion generation failed", stderr.getvalue())
