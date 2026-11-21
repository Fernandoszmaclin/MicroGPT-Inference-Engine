"""Build the optional attention library next to the Python package."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .native import _library_path


def build_native(compiler: str | None = None) -> Path:
    destination = _library_path().resolve()
    source = destination.with_name("attention.c")
    if compiler is None:
        compiler = os.environ.get("CC") or ("cl" if sys.platform == "win32" else "cc")
    executable = shutil.which(compiler)
    if executable is None:
        raise ValueError(f"compiler {compiler!r} was not found on PATH")

    with TemporaryDirectory(prefix="microgpt-", dir=destination.parent) as temp:
        output = Path(temp) / destination.name
        if Path(executable).stem.lower() in {"cl", "clang-cl"}:
            command = [executable, "/nologo", "/O2", "/std:c11", "/LD", str(source), f"/Fe:{output}"]
        else:
            link = "-dynamiclib" if sys.platform == "darwin" else "-shared"
            command = [executable, "-O2", "-std=c11", link]
            if sys.platform != "win32":
                command.append("-fPIC")
            command.extend([str(source), "-o", str(output), "-lm"])
        subprocess.run(command, cwd=temp, check=True)
        output.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", help="compiler executable (defaults to CC or the platform compiler)")
    args = parser.parse_args()
    try:
        print(f"Built {build_native(args.cc)}")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
