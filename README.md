# Radio Fundamentals

Interactive Jupyter notebooks for learning radio and DSP fundamentals from first principles, with an emphasis on waveform intuition, audio demos, and practical RF examples.

## Repository status

This repository is being built out from the implementation plan in [`PLAN.md`](./PLAN.md). The long-term structure is a modular notebook curriculum organized by topic, with shared helpers in `rf_utils.py`.

## Environment

This project uses a local virtual environment at `.venv/`, which is ignored by git.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Local assets

Recordings and IQ captures are intentionally not stored in git. Put local-only inputs under `assets/local/` or another ignored location.

## Roadmap

See [`BACKLOG.md`](./BACKLOG.md) for the execution backlog and [`PLAN.md`](./PLAN.md) for the full curriculum and product scope.
