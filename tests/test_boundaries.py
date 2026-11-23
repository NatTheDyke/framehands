"""framehands imports only itself, FrameNPU and third-party libraries: never the tracker app."""
import ast
import pathlib
import unittest

PKG = pathlib.Path(__file__).resolve().parents[1] / "framehands"
APP = {"handtrack"}
# models run through framenpu.open (the session API) only
KNOWN = {("mercury_kp", "framenpu")}


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


class BoundaryTests(unittest.TestCase):
    def test_no_app_imports(self):
        for p in PKG.glob("*.py"):
            for name in imports(p):
                self.assertNotIn(name.split(".")[0], APP, f"{p.name} imports {name}")

    def test_framenpu_only_through_the_session_api(self):
        found = {(p.stem, n) for p in PKG.glob("*.py") for n in imports(p) if n.split(".")[0] == "framenpu"}
        self.assertEqual(found, KNOWN)


if __name__ == "__main__":
    unittest.main()
