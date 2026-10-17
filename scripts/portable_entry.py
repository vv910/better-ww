"""Runtime entry point for the standalone Windows portable build."""

import importlib
import json
import os
from pathlib import Path
import sys
import traceback


def application_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def configure_portable_runtime():
    app_dir = application_dir()
    os.chdir(app_dir)

    release_path = app_dir / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    version = release.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("release.json must contain a non-empty version")

    import config as project_config

    project_config.version = version
    project_config.config["version"] = version
    project_config.config['gui_title'] = 'Better WW · 3C'
    project_config.config.pop("update_pyappify", None)
    for locale in ("default", "zh_CN"):
        links = project_config.config.get("links", {}).get(locale)
        if links is not None:
            links["github"] = "https://github.com/vv910/better-ww"
            links['download'] = 'https://github.com/vv910/better-ww/releases'
            links['share'] = 'Better WW: https://github.com/vv910/better-ww/releases'
    return app_dir, version


def run_self_test(app_dir, version):
    required_resources = (
        "assets/coco_annotations.json",
        "assets/echo_model/echo.onnx",
        "assets/images/0.png",
        "icons/icon.png",
        "onnxocr/models/ppocrv5/det/det.onnx",
        "onnxocr/models/ppocrv5/rec/rec.onnx",
        "onnxocr/models/ppocrv5/ppocrv5_dict.txt",
        "i18n/zh_CN/LC_MESSAGES/ok.mo",
        "routes/3c/README.md",
        "README.md",
        "LICENSE.txt",
    )
    modules = (
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "ok",
        "numpy",
        "cv2",
        "openvino",
        "onnxocr",
        "src.task.Farm3CEchoTask",
        "src.pathing.MapLocalizer",
        "src.globals",
    )
    missing = [resource for resource in required_resources if not (app_dir / resource).is_file()]
    if missing:
        raise FileNotFoundError("Missing portable resources: " + ", ".join(missing))

    imported = []
    for module in modules:
        importlib.import_module(module)
        imported.append(module)

    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    if QPixmap(str(app_dir / 'icons/icon.png')).isNull():
        raise RuntimeError('Qt could not load the application icon')
    app.processEvents()

    from openvino import Core

    openvino_devices = Core().available_devices
    if not openvino_devices:
        raise RuntimeError("OpenVINO loaded without any available execution devices")

    report = {
        "ok": True,
        "version": version,
        "application_dir": str(app_dir),
        "imports": imported,
        "qt_initialized": True,
        "openvino_devices": openvino_devices,
        "resources": list(required_resources),
    }
    (app_dir / "smoke-test.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


def main():
    app_dir, version = configure_portable_runtime()
    if "--self-test" in sys.argv[1:]:
        run_self_test(app_dir, version)
        return 0
    if len(sys.argv) > 1:
        raise ValueError("Supported argument: --self-test")

    from ok import OK
    from config import config

    OK(config).start()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        (application_dir() / 'startup-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise SystemExit(1)
