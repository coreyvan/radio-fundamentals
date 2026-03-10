# Radio Fundamentals Backlog

## Phase 1 - Repository Foundation

- [x] Move the in-repo virtual environment into `.venv/`
- [x] Replace the venv-generated `.gitignore` with project ignore rules
- [x] Create the section directory scaffold from `PLAN.md`
- [x] Add baseline project files: `README.md`, `requirements.txt`, `rf_utils.py`
- [x] Define local-only asset handling so recordings and IQ captures stay out of git
- [x] Add a reproducible notebook template for future section notebooks

## Phase 2 - Shared DSP Foundation

- [x] Implement the first `rf_utils.py` pass from `PLAN.md`
- [x] Add reusable plotting helpers for waveform, spectrum, and spectrogram views
- [x] Add reusable audio/widget helpers used across notebooks
- [x] Verify utility behavior with short smoke tests or executable examples

## Phase 3 - Decompose Existing Notebook Content

- [x] Extract FFT and spectrum material into `03-frequency-domain/01-fourier-transform.ipynb`
- [x] Extract practical FFT and windowing material into `03-frequency-domain/02-fft-in-practice.ipynb`
- [x] Extract filter concepts into `04-filters/01-filter-concepts.ipynb`
- [x] Extract filter design material into `04-filters/02-filter-design.ipynb`
- [x] Extract AM fundamentals into `05-am-modulation/01-am-fundamentals.ipynb`
- [x] Extract FM fundamentals into `06-fm-modulation/01-fm-fundamentals.ipynb`
- [x] Extract bandwidth/deviation comparison into `06-fm-modulation/02-narrowband-vs-wideband.ipynb`
- [x] Extract pre/de-emphasis and noise demos into `06-fm-modulation/03-pre-emphasis-de-emphasis.ipynb`
- [x] Extract noise and intelligibility demos into `10-noise-and-sensitivity/01-thermal-noise.ipynb`
- [x] Extract SNR comparison material into `10-noise-and-sensitivity/02-snr-and-intelligibility.ipynb`

## Phase 4 - Build the Core Curriculum

- [x] Author section 01 notebooks for signals, superposition, harmonics, and beat frequencies
- [x] Author section 02 notebooks for sampling, Nyquist, aliasing, and reconstruction
- [x] Author section 07 notebooks for the receiver chain and tuning demos
- [x] Author section 08 notebooks for transmitter fundamentals, PLLs, and signaling
- [x] Author section 09 notebooks for antennas, link budgets, and propagation
- [x] Author section 11 notebooks for digital modulation, constellations, BER, and LoRa

## Phase 5 - Practical and Hardware-Oriented Work

- [x] Add prerecorded-file paths for all practical notebooks before any live SDR dependency
- [x] Implement section 12 practical notebooks around FM, APRS, Baofeng, Meshtastic, and POCSAG
- [x] Implement section 13 measurement notebooks around spectrum, deviation, and sensitivity
- [x] Gate RTL-SDR-specific features behind optional dependencies and device checks

## Phase 6 - Cleanup and Validation

- [ ] Add Pi-oriented setup notes for JupyterLab, `ipympl`, `ffmpeg`, and optional SDR packages
- [ ] Smoke-test each notebook with restart-and-run-all discipline
- [ ] Remove the legacy notebooks after their content has been fully decomposed
- [ ] Review the repo for large-file hygiene before the first real commit series
