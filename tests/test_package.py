import unittest

from asr import __version__


class PackageMetadataTest(unittest.TestCase):
    def test_version_is_defined(self) -> None:
        self.assertTrue(__version__)
