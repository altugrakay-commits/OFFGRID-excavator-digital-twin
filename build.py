"""
Cross-platform build script for the C stress solver.

Usage:
    python build.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "c_src" / "stress_solver.c"

SYSTEM = platform.system()


def _run(cmd: list[str], label: str) -> bool:
    print(f"[build] Trying: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"[build] {label}: tool not found on PATH.")
        return False

    if result.returncode != 0:
        print(f"[build] {label}: failed (exit {result.returncode}).")
        if result.stderr:
            print(result.stderr.strip()[:500])
        return False

    print(f"[build] {label}: success.")
    return True


def build_windows() -> bool:
    """Try MinGW-w64 gcc, then MSVC cl.exe."""
    out = ROOT / "libstress.dll"

    # --- MinGW-w64 gcc ---
    if shutil.which("gcc"):
        if _run(
            ["gcc", "-O3", "-shared", "-o", str(out), str(SRC), "-lm"],
            "MinGW gcc",
        ):
            return out.exists()

    # --- MSVC cl.exe (must be run from a Developer Command Prompt) ---
    if shutil.which("cl"):
        # cl writes .dll next to the .c file by default; use /Fe to control
        if _run(
            ["cl", "/LD", "/O2", f"/Fe:{out}", str(SRC)],
            "MSVC cl",
        ):
            return out.exists()

    print(
        "\n[build] No Windows C compiler found.\n"
        "        Install one of:\n"
        "          - MinGW-w64 (via MSYS2, or 'winget install mingw')\n"
        "          - Visual Studio Build Tools (C++ workload)\n"
        "        Then re-run: python build.py\n"
    )
    return False


def build_unix() -> bool:
    out = ROOT / ("libstress.dylib" if SYSTEM == "Darwin" else "libstress.so")
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("[build] No C compiler found (cc/gcc/clang).")
        return False

    flags = ["-O3", "-fPIC", "-shared", "-o", str(out), str(SRC), "-lm"]
    if SYSTEM == "Darwin":
        flags = ["-O3", "-fPIC", "-dynamiclib", "-o", str(out), str(SRC), "-lm"]

    if _run([cc, *flags], cc):
        return out.exists()
    return False


def main() -> int:
    print(f"[build] Platform: {SYSTEM}")
    print(f"[build] Source:   {SRC}")
    if not SRC.exists():
        print(f"[build] ERROR: source not found at {SRC}")
        return 1

    ok = build_windows() if SYSTEM == "Windows" else build_unix()

    if ok:
        print(f"[build] Done. Library is ready.")
        return 0
    print("[build] Build failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())