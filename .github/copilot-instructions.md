# Project Guidelines

## Project Status
- This repository is in an early scaffold stage: currently only `README.md` exists.
- Treat implementation details in the README as planned structure, not guaranteed current state.
- Before making code changes, verify that referenced files and folders actually exist.

## Architecture
- Planned layout (from `README.md`):
  - `src/detect.py`: main inference entry point
  - `src/utils.py`: helper utilities
  - `models/`: model files/weights
  - `data/`: sample inputs
- Keep model-framework decisions explicit (MediaPipe, YOLOv8, PyTorch, TensorFlow). Do not mix framework-specific patterns in the same change unless requested.

## Build And Test
- No canonical build/test automation is defined yet (no `requirements.txt`, test suite, or task runner currently present).
- If setup is needed, follow and align with `README.md` instructions.
- When adding runnable code, also add reproducible setup and verification commands to `README.md`.

## Conventions
- Default language is Python 3.x for implementation work in this project.
- Prefer small, composable modules under `src/` and keep CLI behavior documented.
- For detection CLI work, preserve the README argument shape unless asked to change it: `--source`, optional `--conf`, optional `--save`.
- Keep paths and examples Windows-compatible when possible.

## Documentation
- Source of truth is `README.md` for project intent and usage examples.
- Link to existing docs instead of duplicating long instructions in code comments or new markdown files.
