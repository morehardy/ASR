import unittest
from pathlib import Path

from asr.cli import build_parser, resolve_cli_inputs


class CliParserTest(unittest.TestCase):
    def test_defaults_to_current_directory_when_input_missing(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(resolve_cli_inputs(args.inputs), [Path.cwd()])

    def test_recursive_is_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["./media", "--recursive"])

        self.assertTrue(args.recursive)
