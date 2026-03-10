# Radio Fundamentals

Interactive Jupyter notebooks for learning radio and DSP fundamentals from first principles, with an emphasis on waveform intuition, audio demos, and practical RF examples.

## Environment

This project uses a local virtual environment at `.venv/`, which is ignored by git.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `python -m pip` form matters here because this virtualenv was moved into `.venv/` after initial creation.

## Running JupyterLab

```bash
source .venv/bin/activate
jupyter lab
```

For Raspberry Pi setup, headless JupyterLab configuration, and the sample `systemd` service, see [`PI_SETUP.md`](./PI_SETUP.md).

## Local assets

Recordings and IQ captures are intentionally not stored in git. Put local-only inputs under `assets/local/` or another ignored location.

Recommended capture filenames and formats are documented in [`assets/capture_manifest.md`](./assets/capture_manifest.md).

## Repository layout

- `rf_utils.py`: shared DSP, plotting, widget, and IQ helper functions
- `01-signals-and-waves/` through `13-test-and-measurement/`: modular notebook curriculum
- `scripts/`: notebook generation scripts used to build the curriculum structure
- `tests/`: lightweight validation for the shared utility layer

## Roadmap

See [`BACKLOG.md`](./BACKLOG.md) for the execution backlog and [`PLAN.md`](./PLAN.md) for the full curriculum and product scope.
