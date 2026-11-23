"""Build the C++ geometry core's Python module (framehands/_core*.so) with CMake.

    python tools/build_core.py [--pybind11-dir DIR] [--python EXE]

The module goes next to the Python sources (framehands/); the build tree is a fresh temporary
directory. Needs CMake, a C++17 compiler and pybind11 (pip install pybind11).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pybind11-dir", help="pybind11's CMake directory (default: python -m pybind11 --cmakedir)")
    ap.add_argument("--python", default=sys.executable, help="the Python to build for")
    a = ap.parse_args()
    pdir = a.pybind11_dir or subprocess.check_output([a.python, "-m", "pybind11", "--cmakedir"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="framehands-core-") as build:
        subprocess.check_call(["cmake", "-S", os.path.join(ROOT, "cpp"), "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                               "-Dpybind11_DIR=" + pdir, "-DPython_EXECUTABLE=" + a.python])
        subprocess.check_call(["cmake", "--build", build, "-j", str(os.cpu_count() or 2)])
        built = glob.glob(os.path.join(build, "_core*.so"))
        if len(built) != 1:
            sys.exit("expected one _core module, got %s" % built)
        dest = os.path.join(ROOT, "framehands", os.path.basename(built[0]))
        shutil.copy2(built[0], dest)
        print(dest)


if __name__ == "__main__":
    main()
