# Agent Instructions

## Python

- When running Python commands in this repository, use the local virtual environment if it exists.
- On Windows/PowerShell, prefer `.\.venv\Scripts\python.exe` when present.
- On POSIX shells, prefer `./.venv/bin/python` when present.
- Fall back to `python` only when no local `.venv` interpreter exists.
- Prefer invoking the interpreter directly, for example `.\.venv\Scripts\python.exe -m pytest`, instead of relying on shell activation.

## Screenshot tests

- Use `run_test.py` for image-based task tests so OCR warmup is controlled.
- Run each module inheriting `TaskTestCase` in a separate process; its global capture backend is not reusable after teardown.
- Route formats and runtime limits are documented in `docs/development/echo-farming.md`; setup on another Windows PC is in `docs/development/testing-on-another-pc.md`.
- Portable packaging and release checks are in `docs/development/releases.md`. The builder stages Git-tracked files only; stage new runtime modules before building.
