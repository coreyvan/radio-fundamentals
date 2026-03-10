# Radio Fundamentals — Interactive Learning Tool

## Project Overview

A self-paced, interactive learning tool built as a collection of modular Jupyter notebooks that teach radio and DSP fundamentals from the ground up. The learner can see waveforms, hear audio, drag sliders to change parameters in real time, and eventually work with real RF signals via an RTL-SDR dongle.

The target audience is someone who has programming experience (Python) and an interest in radio (ham, GMRS, Meshtastic) but wants to build deep intuition for the signal processing that happens inside every radio — from the moment you key up a Baofeng to the moment someone hears your voice on the other end.

## Runtime Environment

- **Hardware**: Raspberry Pi 4 (4GB or 8GB), running headless
- **OS**: Raspberry Pi OS Lite 64-bit (aarch64, no desktop environment)
- **Python environment**: Virtual environment at `~/jupyter-dsp`
- **Jupyter server**: JupyterLab, running as a systemd service on port 8888
- **Development workflow**: Code runs on the Pi, browser UI accessed from a Mac on the local network at `http://<pi-hostname>.local:8888`
- **Future hardware**: RTL-SDR dongle plugged into the Pi for real RF capture

### Pi Setup Summary

```bash
# Python environment
python3 -m venv ~/jupyter-dsp
source ~/jupyter-dsp/bin/activate
pip install --upgrade pip
pip install jupyterlab numpy scipy matplotlib ipywidgets ffmpeg-python

# System dependencies
sudo apt update
sudo apt install python3-pip python3-venv ffmpeg

# Jupyter config
jupyter lab --generate-config
# In ~/.jupyter/jupyter_lab_config.py:
#   c.ServerApp.ip = '0.0.0.0'
#   c.ServerApp.open_browser = False
#   c.ServerApp.port = 8888
jupyter lab password

# Systemd service at /etc/systemd/system/jupyter.service
# See setup section below for full unit file
```

### Systemd Unit File

```ini
[Unit]
Description=Jupyter Lab
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/notebooks
ExecStart=/home/pi/jupyter-dsp/bin/jupyter-lab
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable jupyter
sudo systemctl start jupyter
```

## Project Structure

```
radio-fundamentals/
├── README.md                          # This file
├── rf_utils.py                        # Shared utility module (all notebooks import from here)
├── requirements.txt                   # pip dependencies
├── assets/                            # Audio samples, IQ recordings, etc.
│   └── my_voice.m4a                   # User voice recording for modulation demos
│
├── 01-signals-and-waves/
│   ├── 01-what-is-a-signal.ipynb
│   └── 02-adding-signals.ipynb
│
├── 02-sampling-and-digital/
│   ├── 01-analog-to-digital.ipynb
│   └── 02-aliasing.ipynb
│
├── 03-frequency-domain/
│   ├── 01-fourier-transform.ipynb
│   └── 02-fft-in-practice.ipynb
│
├── 04-filters/
│   ├── 01-filter-concepts.ipynb
│   └── 02-filter-design.ipynb
│
├── 05-am-modulation/
│   ├── 01-am-fundamentals.ipynb
│   └── 02-ssb-and-hf.ipynb
│
├── 06-fm-modulation/
│   ├── 01-fm-fundamentals.ipynb
│   ├── 02-narrowband-vs-wideband.ipynb
│   └── 03-pre-emphasis-de-emphasis.ipynb
│
├── 07-receiver-chain/
│   ├── 01-superheterodyne.ipynb
│   └── 02-build-a-receiver.ipynb
│
├── 08-transmitter-fundamentals/
│   ├── 01-oscillators-and-plls.ipynb
│   └── 02-ctcss-and-signaling.ipynb
│
├── 09-antennas-and-propagation/
│   ├── 01-antenna-basics.ipynb
│   ├── 02-propagation.ipynb
│   └── 03-link-budgets.ipynb
│
├── 10-noise-and-sensitivity/
│   ├── 01-thermal-noise.ipynb
│   └── 02-snr-and-intelligibility.ipynb
│
├── 11-digital-modulation/
│   ├── 01-ask-fsk-psk-qam.ipynb
│   ├── 02-constellation-diagrams.ipynb
│   └── 03-lora-chirp-spread-spectrum.ipynb
│
├── 12-practical-projects/
│   ├── 01-fm-receiver-from-iq.ipynb
│   ├── 02-decode-aprs.ipynb
│   ├── 03-analyze-baofeng-tx.ipynb
│   ├── 04-meshtastic-signal-analysis.ipynb
│   └── 05-decode-pocsag.ipynb
│
└── 13-test-and-measurement/
    ├── 01-rtlsdr-spectrum-analyzer.ipynb
    ├── 02-measuring-deviation.ipynb
    └── 03-measuring-sensitivity.ipynb
```

## Shared Utility Module: `rf_utils.py`

Every notebook imports from this module. Promote functions here when they're reused across notebooks. Start with these and grow as needed:

```python
"""rf_utils.py — Shared DSP utilities for radio-fundamentals notebooks."""

import numpy as np
from scipy import signal, fft
from scipy.signal import hilbert, butter, sosfilt, lfilter
from scipy.io import wavfile
import subprocess, tempfile, os


# ── Signal Generation ──────────────────────────────────────────────

def generate_tone(freq, duration, fs, amplitude=1.0, phase=0.0):
    """Generate a cosine tone."""
    t = np.arange(0, duration, 1/fs)
    return t, amplitude * np.cos(2 * np.pi * freq * t + phase)


def normalize(x):
    """Normalize signal to [-1, 1]."""
    return x / (np.max(np.abs(x)) + 1e-12)


# ── Spectral Analysis ─────────────────────────────────────────────

def power_spectrum(sig, fs, nfft=None):
    """Compute single-sided power spectrum in dB."""
    if nfft is None:
        nfft = len(sig)
    S = fft.fft(sig, n=nfft)
    S_mag = np.abs(S[:nfft // 2]) / nfft
    S_db = 20 * np.log10(S_mag + 1e-12)
    freqs = np.linspace(0, fs / 2, nfft // 2)
    return freqs, S_db


# ── Modulation ─────────────────────────────────────────────────────

def am_modulate(message, carrier_freq, fs, mod_index=0.8):
    """AM modulate a message signal."""
    t = np.arange(len(message)) / fs
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    return (1 + mod_index * message) * carrier


def fm_modulate(message, carrier_freq, fs, freq_dev=2500):
    """FM modulate a message signal."""
    t = np.arange(len(message)) / fs
    phase_integral = np.cumsum(message) / fs
    return np.cos(2 * np.pi * carrier_freq * t + 2 * np.pi * freq_dev * phase_integral)


# ── Demodulation ───────────────────────────────────────────────────

def am_demodulate(rx_signal, fs, audio_cutoff=3500):
    """AM envelope detector: rectify + lowpass."""
    rectified = np.abs(rx_signal)
    sos_lp = butter(5, audio_cutoff, btype='low', fs=fs, output='sos')
    envelope = sosfilt(sos_lp, rectified)
    envelope = envelope - np.mean(envelope)
    return normalize(envelope)


def fm_demodulate(rx_signal, fs, audio_cutoff=3500):
    """FM demodulation via arctan discriminator."""
    analytic = hilbert(rx_signal)
    inst_phase = np.unwrap(np.angle(analytic))
    demod = np.diff(inst_phase) * fs / (2 * np.pi)
    demod = np.append(demod, demod[-1])
    demod = demod - np.mean(demod)
    sos_lp = butter(5, audio_cutoff, btype='low', fs=fs, output='sos')
    demod = sosfilt(sos_lp, demod)
    return normalize(demod)


# ── Noise ──────────────────────────────────────────────────────────

def add_awgn(signal_data, snr_db):
    """Add white Gaussian noise at a specified SNR (dB)."""
    sig_power = np.mean(signal_data ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal_data))
    return signal_data + noise, noise


def add_impulse_noise(signal_data, fs, rate=50, amplitude=5.0, seed=42):
    """Add random impulse spikes (lightning/ignition noise)."""
    rng = np.random.RandomState(seed)
    noise = np.zeros_like(signal_data)
    num_spikes = int(rate * len(signal_data) / fs)
    spike_locs = rng.randint(0, len(signal_data), num_spikes)
    for loc in spike_locs:
        width = rng.randint(3, 20)
        end = min(loc + width, len(signal_data))
        polarity = rng.choice([-1, 1])
        noise[loc:end] = polarity * amplitude * rng.uniform(0.5, 1.0)
    return signal_data + noise, noise


# ── Emphasis ───────────────────────────────────────────────────────

def pre_emphasis(audio, fs, tau=750e-6):
    """Pre-emphasis filter (boost highs). tau=750µs for land mobile."""
    alpha = np.exp(-1 / (fs * tau))
    return lfilter([1, -alpha], [1], audio)


def de_emphasis(audio, fs, tau=750e-6):
    """De-emphasis filter (cut highs). Inverse of pre-emphasis."""
    alpha = np.exp(-1 / (fs * tau))
    return lfilter([1], [1, -alpha], audio)


# ── Audio I/O ──────────────────────────────────────────────────────

def load_audio(filepath):
    """Load any audio format. Uses scipy for wav, ffmpeg for everything else."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.wav':
        rate, data = wavfile.read(filepath)
        return rate, data
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        subprocess.run([
            'ffmpeg', '-y', '-i', filepath,
            '-ac', '1', '-ar', '44100', '-sample_fmt', 's16',
            tmp.name
        ], check=True, capture_output=True)
        rate, data = wavfile.read(tmp.name)
        os.unlink(tmp.name)
        return rate, data


# ── Propagation ────────────────────────────────────────────────────

def fspl_db(d_km, f_mhz):
    """Free-space path loss in dB."""
    return 20 * np.log10(d_km) + 20 * np.log10(f_mhz) + 32.44
```

## Section Breakdown — Content and Interactive Elements

### 01 — Signals and Waves

**What it covers**: What is a signal? Sine waves as the atomic unit of DSP. Amplitude, frequency, phase. Adding signals together (superposition). Generating tones and listening to them.

**Key concepts**: s(t) = A·cos(2πft + φ), constructive/destructive interference, harmonics, timbre vs pitch.

**Interactive elements**:
- Sliders for frequency (20 Hz–4000 Hz), amplitude, and phase with real-time waveform plot AND audio playback via `IPython.display.Audio`
- "Build a chord" widget: 3–4 frequency sliders, each with amplitude control, hear the sum and see both time and frequency domain update live
- Visual showing how two signals of nearby frequency create a beat pattern — slider for frequency separation

**Audio payoff**: Hear how adding harmonics builds complex timbres from simple sine waves. Hear beat frequencies emerge as you bring two tones close together.

---

### 02 — Sampling and the Digital World

**What it covers**: Analog to digital conversion. Sample rate. The Nyquist-Shannon sampling theorem. Aliasing — what it is, why it happens, what it sounds like.

**Key concepts**: fs > 2·fmax, aliasing as frequency folding, quantization (briefly), reconstruction.

**Interactive elements**:
- Sample rate slider on a fixed audio signal — hear aliasing artifacts appear as you drop below Nyquist
- Visual: continuous sine wave with sample points overlaid, showing how the same set of samples could represent multiple analog frequencies (the aliasing visual)
- "Staircase" reconstruction visual showing how DAC output approximates the original

**Audio payoff**: The moment you slide the sample rate below 2x the signal frequency and hear the pitch fold down instead of up — that's aliasing made visceral.

---

### 03 — Frequency Domain and FFT

**What it covers**: Time domain vs frequency domain as dual representations. The Fourier Transform as a correlation operation. DFT, FFT (Cooley-Tukey). Frequency resolution (fs/N). Windowing and spectral leakage. Spectrograms (STFT).

**Key concepts**: X(f) = ∫x(t)·e^(-j2πft)dt, discrete version X[k] = Σ x[n]·e^(-j2πkn/N), Nyquist, frequency bins = fs/N spacing, Hann/Hamming/Blackman windows.

**Interactive elements**:
- "Frequency domain synthesizer": sliders for 8–10 frequency components (amplitude + frequency each), see time domain waveform and spectrum update simultaneously. Drag a slider and watch a spike appear/move in the spectrum while the waveform reshapes.
- Window function selector (rectangular, Hann, Hamming, Blackman) with live spectrum showing how spectral leakage changes
- FFT length slider showing resolution tradeoff
- Live spectrogram of microphone input (stretch goal — requires `pyaudio` or similar)

---

### 04 — Filters

**What it covers**: What filters do (pass/block frequency ranges). Lowpass, highpass, bandpass, notch. FIR vs IIR. Filter design parameters (order, cutoff, ripple, rolloff). Butterworth, Chebyshev, Bessel, elliptic. Phase response and group delay. The radio audio bandpass (300–3000 Hz).

**Key concepts**: Transfer function H(f), passband/stopband/transition band, -3dB point, filter order, poles and zeros (conceptual).

**Interactive elements**:
- Filter designer: dropdowns for filter type (LP/HP/BP/notch) and family (Butterworth/Chebyshev/etc.), sliders for cutoff and order. Live magnitude/phase response plot.
- Apply the designed filter to a voice recording — hear the result change as you drag cutoff frequency
- Visual showing what "order" means: overlay 2nd, 4th, 8th order Butterworth on the same plot

**Audio payoff**: Drag the lowpass cutoff from 4000 Hz down to 500 Hz and hear your voice go from clear to muffled to unintelligible. Then switch to bandpass at 300–3000 Hz and hear "radio voice."

---

### 05 — AM Modulation

**What it covers**: Why modulation exists (baseband doesn't radiate, frequency division multiplexing). AM math: s(t) = [1 + m·x(t)]·Ac·cos(2πfct). Sidebands (USB, LSB), carrier. Modulation index. Overmodulation and splatter. Envelope detection. DSB-SC, SSB — why hams use SSB on HF.

**Key concepts**: Multiplication in time = frequency shift (modulation property of FT), m = ΔA/Ac, bandwidth = 2·fmax (DSB) or fmax (SSB), the "wasted carrier" problem.

**Interactive elements**:
- Modulation index slider (0→1.5): see envelope, spectrum, and hear demodulated audio. Watch overmodulation distortion appear above m=1.
- Side-by-side DSB vs SSB spectrum comparison
- AM broadcast band simulation: three "stations" at different carrier frequencies, tune across them with a slider, hear each one get selected by the IF filter

**Audio payoff**: Hear your voice AM-modulated and demodulated. Hear overmodulation distortion. Hear the difference between DSB and SSB.

---

### 06 — FM Modulation

**What it covers**: FM math: s(t) = Ac·cos(2πfct + 2πkf∫m(τ)dτ). Frequency deviation, modulation index (β = Δf/fmsg). Bessel function sidebands. Carson's rule BW ≈ 2(Δf + fmax). Narrowband vs wideband FM. The FM noise advantage. Pre-emphasis and de-emphasis (750µs for NA land mobile). GMRS channel parameters (±2.5 kHz narrow, ±5 kHz wide, 12.5/25 kHz spacing).

**Key concepts**: Constant envelope, information in frequency not amplitude, Bessel functions Jn(β), carrier null at β≈2.405, FM improvement factor, noise triangle.

**Interactive elements**:
- Deviation slider with live spectrum (see Bessel sidebands rearrange), time domain (see cycle density change), instantaneous frequency plot, and audio playback
- AM vs FM noise comparison: shared SNR slider, hear both simultaneously. This is the single most impactful demo.
- Pre-emphasis/de-emphasis toggle at low SNR — hear the hiss reduction
- Impulse noise demo: same spikes added to both AM and FM, hear the difference
- FM capture effect: two signals, slider for relative power, hear one dominate

**Audio payoff**: This section has the most audio demos. The SNR sweep (30→0 dB) with AM and FM side by side is the centerpiece of the whole learning tool.

---

### 07 — The Receiver Chain

**What it covers**: Superheterodyne architecture end-to-end. RF front end (BPF + LNA). Mixers and frequency conversion — why multiplication = frequency shift matters here. IF filtering and selectivity. AGC. FM discriminator. Squelch (carrier and tone). The complete signal path from antenna to speaker.

**Key concepts**: LO + RF → IF (sum/difference), image frequency problem, IF bandwidth = selectivity, noise figure, sensitivity.

**Interactive elements**:
- Block diagram with clickable stages — click a stage, see the signal at that point in both time and frequency domain
- "Tune the radio" widget: simulated RF band with 3–4 signals at different frequencies, LO frequency slider that shifts the IF window across the band. See the IF filter select one signal, hear the demodulated audio change as you tune.
- Mixer math visualizer: two input frequencies, see the four output products, watch the IF filter keep one

---

### 08 — Transmitter Fundamentals

**What it covers**: Oscillators (crystal, VCO). PLL synthesizers — how they lock a VCO to a reference. Power amplifiers (class A/B/C tradeoffs for FM). Harmonic filtering. Spectral mask compliance. CTCSS and DCS tone signaling. The composite baseband signal (voice + CTCSS).

**Key concepts**: PLL loop (reference → phase detector → loop filter → VCO → divider → feedback), harmonics at n·f0, PA efficiency vs linearity tradeoff, CTCSS frequencies (67.0–254.1 Hz).

**Interactive elements**:
- PLL lock-in animation: watch the VCO frequency converge on the reference, slider for loop bandwidth showing fast vs slow lock
- Harmonic spectrum before/after filtering: slider for number of filter poles
- CTCSS tone selector: see it appear in the baseband spectrum below 300 Hz, hear it with and without the high-pass filter that normally strips it

---

### 09 — Antennas and Propagation

**What it covers**: How antennas radiate (current → time-varying E/H fields → EM wave). Impedance and SWR. Matching. Radiation patterns: isotropic, dipole, ground plane, Yagi, collinear. Gain (dBi, dBd). Polarization. Free space path loss and the Friis equation. Link budgets. Fresnel zones. Line of sight, diffraction, multipath, ground reflections. VHF/UHF propagation characteristics.

**Key concepts**: Pr = Pt·Gt·Gr·(λ/4πd)², SWR = (1+|Γ|)/(1-|Γ|), Fresnel zone radius, knife-edge diffraction, inverse square law.

**Interactive elements**:
- Link budget calculator: sliders for TX power, TX antenna gain, RX antenna gain, frequency, distance. Shows received power, margin above sensitivity, and a "would this link work?" verdict. Presets for common scenarios (Baofeng-to-Baofeng, Baofeng-to-repeater, Meshtastic node-to-node).
- Radiation pattern viewer: select antenna type, see 2D or 3D pattern. Compare rubber duck vs half-wave vs Yagi.
- Fresnel zone visualizer: distance slider, see the zone width at midpoint, overlay terrain profile
- Path loss comparison across frequencies (HF, VHF, UHF, 915 MHz, 2.4 GHz)

---

### 10 — Noise and Sensitivity

**What it covers**: Thermal noise (kTB). Noise figure and cascaded noise figure (Friis formula for noise). Receiver sensitivity. SNR and SINAD. What noise sounds like across modulation types. The 12 dB SINAD standard for FM.

**Key concepts**: Pn = kTB = -174 dBm/Hz at room temp, NF in dB, sensitivity = kTB + NF + required SNR, SINAD, MDS (minimum discernible signal).

**Interactive elements**:
- SNR slider with simultaneous AM/FM audio playback — the definitive "hear the difference" demo
- Noise figure cascading calculator: chain of LNA → cable loss → mixer → IF amp, see how front-end NF dominates
- Sensitivity calculator: plug in NF, bandwidth, required SINAD → get sensitivity in dBm

---

### 11 — Digital Modulation

**What it covers**: Why digital (error correction, encryption, efficiency). ASK, FSK, PSK, QAM — each as a way of mapping bits to signal parameters. Constellation diagrams. Symbol rate vs bit rate. Bit error rate (BER). Eye diagrams. Spread spectrum concepts. LoRa chirp spread spectrum (directly relevant to Meshtastic).

**Key concepts**: Bits per symbol = log2(M), Nyquist bandwidth, Shannon capacity C = B·log2(1 + SNR), CSS chirps encode data in start frequency, spreading factor = chirp duration.

**Interactive elements**:
- Constellation diagram with noise slider: watch QPSK/16-QAM points blur and cross decision boundaries. Show BER increasing.
- Eye diagram for BPSK/QPSK with noise and ISI
- LoRa chirp visualizer: spectrogram showing chirps, spreading factor selector (SF7–SF12), see chirps get longer and narrower in bandwidth. Explain the range/datarate tradeoff.
- FSK demodulator applied to real AFSK (Bell 202) tones — the basis for APRS

---

### 12 — Practical Projects

Each notebook is standalone. These tie everything together with real-world applications.

**01 — FM Receiver from IQ Samples**: Load RTL-SDR IQ capture of a GMRS or FM broadcast signal. Downconvert, filter, FM-demodulate, play audio. Build a complete software receiver.

**02 — Decode APRS**: Capture 144.39 MHz with RTL-SDR (or use a provided recording). FM demodulate → AFSK demodulate → AX.25 frame decode → parse position reports. Display on a map.

**03 — Analyze Baofeng TX**: Record your Baofeng's transmission via RTL-SDR. Measure actual deviation, spectral occupancy, CTCSS tone level, harmonic content. Compare to spec.

**04 — Meshtastic Signal Analysis**: Capture LoRa packets at 915 MHz. Visualize chirps in spectrogram. Measure signal strength at different distances. Compare spreading factors.

**05 — Decode POCSAG Pager Traffic**: Capture pager signals on UHF. FSK demodulate. Decode POCSAG frames. Simple intro to digital protocol decoding.

---

### 13 — Test and Measurement

**01 — RTL-SDR as Spectrum Analyzer**: Turn the RTL-SDR into a swept spectrum analyzer. Measure spurious emissions, band occupancy, interference sources.

**02 — Measuring Deviation**: Transmit a known tone, measure FM deviation from the received signal. Verify your Baofeng is within spec.

**03 — Measuring Sensitivity**: Inject a calibrated signal (or use relative measurements), determine the weakest signal your receiver can demodulate at 12 dB SINAD.

---

## Key Python Dependencies

```
# requirements.txt
jupyterlab
numpy
scipy
matplotlib
ipywidgets          # interactive sliders, dropdowns, toggles
# Optional / later:
# pyrtlsdr          # RTL-SDR hardware interface
# pyaudio           # live microphone input
# folium            # map display for APRS project
# scikit-dsp-comm   # pre-built comms DSP functions
```

## Interactive Widget Patterns

All interactive notebooks use `ipywidgets` with `%matplotlib widget` for live-updating plots. The standard pattern:

```python
%matplotlib widget
import ipywidgets as widgets
from IPython.display import Audio, display

def update_plot(freq=1000, amplitude=1.0):
    ax.clear()
    t, sig = generate_tone(freq, 0.01, 44100, amplitude)
    ax.plot(t * 1000, sig)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude')
    fig.canvas.draw_idle()

fig, ax = plt.subplots(figsize=(10, 3))
widgets.interact(update_plot,
    freq=widgets.FloatSlider(min=100, max=4000, step=50, value=1000, description='Freq (Hz)'),
    amplitude=widgets.FloatSlider(min=0, max=1, step=0.05, value=1.0, description='Amplitude'))
```

For audio playback that updates with sliders, use `widgets.Output()`:

```python
audio_out = widgets.Output()

def update_with_audio(freq=1000):
    t, sig = generate_tone(freq, 1.0, 44100)
    # Update plot...
    with audio_out:
        audio_out.clear_output()
        display(Audio(sig, rate=44100))

display(audio_out)
```

## Notebook Conventions

Each notebook follows this structure:

1. **Title cell** (Markdown): Section name, one-paragraph overview of what we'll learn
2. **Setup cell** (Code): `from rf_utils import *` plus any section-specific imports, matplotlib config, `%matplotlib widget` if interactive
3. **Concept cells** (Markdown + Code alternating): Explain a concept in Markdown (with LaTeX math), immediately followed by code that demonstrates it
4. **Interactive cells**: Widgets that let the learner explore parameter spaces
5. **Audio cells**: `IPython.display.Audio` for listening to signals
6. **"What to try" cells** (Markdown): Suggestions for experimentation — "change X to Y and notice what happens to Z"
7. **"Key takeaway" cell** (Markdown): One-paragraph summary at the end

### Notebook header template

```python
# Cell 1 — always first
import sys
sys.path.append('..')  # so we can import rf_utils from the project root
from rf_utils import *
from IPython.display import Audio, display, Markdown
import ipywidgets as widgets
%matplotlib widget

# Section-specific imports below
```

## Key Math Reference

These equations appear throughout the notebooks:

### Fourier Transform
$$X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi ft} dt$$

### DFT
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

### AM Modulation
$$s_{AM}(t) = [1 + m \cdot x(t)] \cdot A_c \cos(2\pi f_c t)$$

### FM Modulation
$$s_{FM}(t) = A_c \cos\left(2\pi f_c t + 2\pi k_f \int_0^t m(\tau) d\tau \right)$$

### FM Modulation Index
$$\beta = \frac{\Delta f}{f_{msg}}$$

### Carson's Rule
$$BW \approx 2(\Delta f + f_{max})$$

### Bessel Function Sidebands
$$s_{FM}(t) = A_c \sum_{n=-\infty}^{\infty} J_n(\beta) \cos(2\pi (f_c + n f_{msg}) t)$$

### Free Space Path Loss
$$FSPL_{dB} = 20\log_{10}(d_{km}) + 20\log_{10}(f_{MHz}) + 32.44$$

### Friis Transmission Equation
$$P_r = P_t G_t G_r \left(\frac{\lambda}{4\pi d}\right)^2$$

### Thermal Noise Power
$$P_n = kTB \quad \text{(= -174 dBm/Hz at 290K)}$$

### Shannon Capacity
$$C = B \log_2(1 + SNR)$$

## Reference Resources

- **PySDR** (pysdr.org) — Free online textbook: DSP + SDR + wireless comms using Python/numpy/scipy. The natural next step from these notebooks.
- **Think DSP** by Allen Downey (greenteapress.com/wp/think-dsp) — Free, programmer-first intro to DSP with Jupyter notebooks. Good complement to PySDR.
- **The Scientist and Engineer's Guide to DSP** (dspguide.com) — Free classic. Best prose explanations of DSP fundamentals.
- **ARRL Antenna Book** (25th edition) — The definitive antenna design reference. Not free (~$50) but worth it.
- **ARRL Handbook** — Comprehensive amateur radio electronics reference.
- **Semtech AN1200.22** — LoRa modulation basics at the signal processing level.
- **GNU Radio** (gnuradio.org) — Visual flowgraph-based SDR framework. Good for real-time processing.
- **dsprelated.com** — Forums, articles, and courses from practitioners.

## GMRS-Specific Parameters (for realistic simulations)

| Parameter | Narrowband | Wideband |
|---|---|---|
| Channel spacing | 12.5 kHz | 25 kHz |
| Max deviation | ±2.5 kHz | ±5.0 kHz |
| Audio bandpass | 300–3000 Hz | 300–3000 Hz |
| Pre-emphasis τ | 750 µs | 750 µs |
| TX power (handheld) | up to 5W | up to 5W |
| TX power (mobile/base) | up to 50W | up to 50W |
| Frequency range | 462/467 MHz | 462/467 MHz |
| Repeater offset | +5 MHz | +5 MHz |
| CTCSS range | 67.0–254.1 Hz | 67.0–254.1 Hz |

## Existing Notebooks

Two notebooks have already been created as starting points:

1. **dsp_radio_fundamentals.ipynb** — Covers baseband audio, AM/FM modulation and spectrum, Bessel functions, FM demodulation, CTCSS analysis, filter design, spectrograms, and link budgets. All visual, no audio playback.

2. **am_vs_fm_audio_demo.ipynb** — Loads a voice recording, AM/FM modulates it, adds noise (Gaussian and impulse), demodulates, and plays audio for A/B comparison. Covers deviation comparison, pre/de-emphasis, FM capture effect, and a full SNR degradation sweep. Supports .wav/.m4a/.mp3 input via ffmpeg.

These can be decomposed into the modular section structure above, with their utility functions promoted to `rf_utils.py`.

## Development Notes

- **Pi 4 performance**: numpy/scipy FFTs on arrays up to ~1M samples are fine. `scipy.signal.hilbert` (used in FM demod) is the heaviest operation — process in chunks for clips longer than ~10 seconds at 96 kHz.
- **Matplotlib on Pi**: `%matplotlib widget` requires `ipympl` (`pip install ipympl`). Static plots (`%matplotlib inline`) work without it but aren't interactive.
- **Audio playback**: `IPython.display.Audio` renders an HTML5 audio player in the browser. The audio data is base64-encoded and sent to the browser, so playback happens on the Mac, not the Pi. This means audio works perfectly over the network with zero latency concerns.
- **File management**: Upload voice recordings and IQ captures through JupyterLab's file browser UI, or `scp` them from the Mac.
- **RTL-SDR future**: When ready, `pip install pyrtlsdr`. The Pi handles RTL-SDR at up to ~2.4 Msps for capture. For real-time processing, keep sample rates under ~1 MHz for complex pipelines.