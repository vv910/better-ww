"""Build a self-contained Windows onedir release using PyInstaller."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIRS = ("assets", "icons", "i18n", "routes", "readme", "docs")
SOURCE_DIRS = ("src",)
ROOT_FILES = ("README.md", "LICENSE.txt")
VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def tracked_paths():
    result = subprocess.run(
        ["git", "ls-files", "--recurse-submodules", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [PurePosixPath(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def is_portable_path(path):
    parts = path.parts
    if not parts or any(part in {"tests", "test", "__pycache__"} for part in parts):
        return False
    if parts[0] in RESOURCE_DIRS or parts[0] in SOURCE_DIRS:
        return True
    if path.as_posix() == "scripts/portable_entry.py":
        return True
    return len(parts) == 1 and (parts[0] in {"config.py"} or parts[0] in ROOT_FILES)


def stage_sources(stage, version):
    stage.mkdir(parents=True, exist_ok=True)
    copied = []
    for relative in tracked_paths():
        if not is_portable_path(relative):
            continue
        source = ROOT.joinpath(*relative.parts)
        if not source.is_file():
            continue
        destination = stage.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())

    for package in SOURCE_DIRS:
        if not (stage / package).is_dir():
            raise FileNotFoundError(f"Expected tracked runtime package: {package}")
    for resource in RESOURCE_DIRS:
        if not (stage / resource).is_dir():
            raise FileNotFoundError(f"Expected tracked runtime resource directory: {resource}")
    for required in ("config.py", "README.md", "LICENSE.txt"):
        if not (stage / required).is_file():
            raise FileNotFoundError(f"Expected tracked release file: {required}")

    entry_source = stage / "scripts" / "portable_entry.py"
    if not entry_source.is_file():
        raise FileNotFoundError("scripts/portable_entry.py must be tracked in Git before building")
    shutil.copy2(entry_source, stage / "portable_entry.py")
    (stage / "release.json").write_text(
        json.dumps({"version": version}, indent=2) + "\n", encoding="utf-8"
    )
    return copied


def run_pyinstaller(stage, dist_root, build_root):
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--contents-directory",
        ".",
        "--name",
        "BetterWW",
        "--distpath",
        str(dist_root),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root / "spec"),
        "--icon",
        str(stage / "icons" / "icon.ico"),
    ]

    # Task and UI modules are referenced by configuration strings and registries,
    # so list every tracked application module for PyInstaller explicitly.
    module_names = set()
    for package in (*SOURCE_DIRS, "ok"):
        package_root = stage / package
        if package == "ok":
            spec = importlib.util.find_spec("ok")
            if spec is None or not spec.submodule_search_locations:
                raise ImportError("The installed runtime dependencies do not provide package 'ok'.")
            package_root = Path(next(iter(spec.submodule_search_locations)))
        if not package_root.is_dir():
            continue
        for module_path in package_root.rglob("*.py"):
            if "__pycache__" in module_path.parts:
                continue
            relative = module_path.relative_to(package_root).with_suffix("")
            suffix = ".".join(relative.parts)
            if suffix.endswith(".__init__"):
                suffix = suffix[: -len(".__init__")]
            module_names.add(package if not suffix else f"{package}.{suffix}")
    for module_name in sorted(module_names):
        command.extend(("--hidden-import", module_name))

    # These packages carry native DLLs, Qt plugins, OCR models, or OpenVINO plugins.
    for package in (
        "qfluentwidgets",
        "cv2",
        "openvino",
        "onnxocr",
        "comtypes",
        "pycaw",
        "pynput",
        "pydirectinput",
        "shapely",
    ):
        command.extend(("--collect-all", package))
    command.extend(("--collect-data", "ok"))

    # Keep staged Python trees beside the frozen packages too. A few optional UI
    # features resolve web/static files from a module's __file__ path at runtime.
    for name in (*RESOURCE_DIRS, *SOURCE_DIRS):
        command.extend(("--add-data", f"{stage / name};{name}"))
    command.extend(("--add-data", f"{stage / 'release.json'};."))
    command.extend(("--add-data", f"{stage / 'README.md'};."))
    command.extend(("--add-data", f"{stage / 'LICENSE.txt'};."))
    command.append(str(stage / "portable_entry.py"))

    # Do not resolve native DLLs from unrelated programs on the developer's PATH
    # (for example Poppler's versioned ICU conflicts with Qt's Windows ICU API).
    build_env = os.environ.copy()
    build_env.pop('PYTHONPATH', None)
    windows = Path(os.environ['SystemRoot'])
    build_env['PATH'] = os.pathsep.join(map(str, (
        Path(sys.executable).parent, Path(sys.base_prefix), windows / 'System32', windows,
    )))
    subprocess.run(command, cwd=stage, env=build_env, check=True)


def write_archive(app_dir, dist_root, version):
    archive = dist_root / f"BetterWW-{version}-windows-x64.zip"
    digest_file = dist_root / f"BetterWW-{version}-windows-x64.sha256"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(app_dir.rglob("*")):
            if path.is_file():
                output.write(path, (Path("BetterWW") / path.relative_to(app_dir)).as_posix())
    with archive.open('rb') as packaged:
        digest = hashlib.file_digest(packaged, 'sha256').hexdigest()
    digest_file.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, digest_file


def main():
    if os.name != "nt":
        raise SystemExit("Portable Windows release builds must run on Windows.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release tag, for example v1.2.3")
    args = parser.parse_args()
    if not VERSION_PATTERN.fullmatch(args.version):
        raise SystemExit("--version must be a tag like v1.2.3 or v1.2.3-beta.1")

    dist_root = ROOT / "dist"
    app_dir = dist_root / "BetterWW"
    build_root = ROOT / "build" / "portable"
    dist_root.mkdir(parents=True, exist_ok=True)
    for generated in (app_dir, build_root):
        resolved = generated.resolve()
        if ROOT not in resolved.parents or generated.is_symlink():
            raise ValueError(f'Build output must stay inside this checkout: {generated}')
        if generated.exists():
            shutil.rmtree(generated)
    build_root.mkdir(parents=True, exist_ok=True)
    stage = build_root / "stage"

    staged_paths = stage_sources(stage, args.version)
    run_pyinstaller(stage, dist_root, build_root)
    required_exe = app_dir / "BetterWW.exe"
    if not required_exe.is_file():
        raise FileNotFoundError(f"PyInstaller did not produce {required_exe}")
    for relative in RESOURCE_DIRS:
        if not (app_dir / relative).exists():
            raise FileNotFoundError(f"PyInstaller output is missing {relative}")
    archive, digest_file = write_archive(app_dir, dist_root, args.version)
    print(f"Staged {len(staged_paths)} tracked files")
    print(f"Portable directory: {app_dir}")
    print(f"Archive: {archive}")
    print(f"SHA256: {digest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
