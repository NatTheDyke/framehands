"""framehands.__version__ is the package version pyproject.toml declares (0.2.1 shipped with a stale one)."""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class VersionTests(unittest.TestCase):
    def test_module_version_matches_pyproject(self):
        import framehands

        declared = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(), re.M)[1]
        self.assertEqual(framehands.__version__, declared)


if __name__ == "__main__":
    unittest.main()
