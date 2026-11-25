"""framehands imports only itself, FrameNPU and third-party libraries: never the tracker app."""
import ast
import pathlib
import re
import sys
import tomllib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "framehands"
# the distribution of an imported package, where the names differ
DISTRIBUTION = {"cv2": "opencv-python-headless"}
# packages only some modules need: the extra that declares each
EXTRAS = {"onnxruntime": "cpu", "mediapipe": "mediapipe"}
APP = {"handtrack"}
# models run through framenpu.open (the session API) only
KNOWN = {("mercury_kp", "framenpu")}


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def declared():
    """(required, by extra) distribution names of pyproject.toml."""
    def names(requirements):
        return {re.match(r"[\w.-]+", r)[0].lower() for r in requirements}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return names(project["dependencies"]), {k: names(v) for k, v in project["optional-dependencies"].items()}


class BoundaryTests(unittest.TestCase):
    def test_every_third_party_import_is_declared(self):
        required, extras = declared()
        for p in PKG.rglob("*.py"):
            for name in {n.split(".")[0] for n in imports(p)} - set(sys.stdlib_module_names) - {"framehands"}:
                where = extras.get(EXTRAS[name], set()) if name in EXTRAS else required
                self.assertIn(DISTRIBUTION.get(name, name), where, f"{p.name} imports {name}, undeclared")

    def test_no_app_imports(self):
        for p in PKG.glob("*.py"):
            for name in imports(p):
                self.assertNotIn(name.split(".")[0], APP, f"{p.name} imports {name}")

    def test_framenpu_only_through_the_session_api(self):
        found = {(p.stem, n) for p in PKG.glob("*.py") for n in imports(p) if n.split(".")[0] == "framenpu"}
        self.assertEqual(found, KNOWN)


if __name__ == "__main__":
    unittest.main()
