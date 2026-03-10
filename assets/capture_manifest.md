# Local Capture Manifest

Phase 5 notebooks look for optional local-only files under `assets/local/`. These files are ignored by git and are not required for the notebooks to run; each notebook falls back to synthetic demo data if its local capture is missing.

Suggested filenames:

- `assets/local/fm_receiver_iq.npz`
- `assets/local/aprs_afsk_iq.npz`
- `assets/local/baofeng_tx_iq.npz`
- `assets/local/meshtastic_iq.npz`
- `assets/local/pocsag_iq.npz`
- `assets/local/spectrum_sweep_iq.npz`
- `assets/local/deviation_test_iq.npz`
- `assets/local/sensitivity_test_iq.npz`

Recommended `.npz` structure:

- `iq`: complex64 or complex128 numpy array of IQ samples
- `sample_rate`: scalar sample rate in Hz

If you later use an RTL-SDR directly, the notebooks also probe for `pyrtlsdr` and a usable device, but they do not require live hardware for the default flow.
