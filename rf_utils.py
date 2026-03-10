"""Shared DSP helpers for the Radio Fundamentals notebook set."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
from typing import Iterable

import numpy as np
from scipy import fft as scipy_fft
from scipy import signal
from scipy.io import wavfile
from scipy.signal import butter, hilbert, lfilter, sosfilt

EPSILON = 1e-12


def generate_tone(
    freq: float,
    duration: float,
    fs: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cosine tone."""
    t = np.arange(0, duration, 1 / fs, dtype=np.float64)
    sig = amplitude * np.cos((2 * np.pi * freq * t) + phase)
    return t, sig


def ensure_mono(data: np.ndarray) -> np.ndarray:
    """Convert multichannel audio to mono by averaging channels."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1)


def normalize(x: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Normalize a signal to a target peak magnitude."""
    arr = np.asarray(x, dtype=np.float64)
    max_mag = np.max(np.abs(arr)) if arr.size else 0.0
    if max_mag <= EPSILON:
        return np.zeros_like(arr, dtype=np.float64)
    return peak * arr / max_mag


def resample_signal(signal_data: np.ndarray, src_fs: float, dst_fs: float) -> np.ndarray:
    """Resample a signal to a new sample rate."""
    if src_fs <= 0 or dst_fs <= 0:
        raise ValueError("Sample rates must be positive.")
    if len(signal_data) == 0 or src_fs == dst_fs:
        return np.asarray(signal_data, dtype=np.float64)
    num_samples = int(round(len(signal_data) * dst_fs / src_fs))
    if num_samples <= 0:
        raise ValueError("Resample would produce an empty signal.")
    return signal.resample(np.asarray(signal_data, dtype=np.float64), num_samples)


def power_spectrum(
    sig: np.ndarray,
    fs: float,
    nfft: int | None = None,
    window: str | Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a single-sided magnitude spectrum in dB."""
    arr = np.asarray(sig, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("power_spectrum expects a 1D signal.")
    if len(arr) == 0:
        raise ValueError("power_spectrum requires at least one sample.")
    nfft = nfft or len(arr)
    if nfft <= 0:
        raise ValueError("nfft must be positive.")

    window_arr = np.ones(len(arr), dtype=np.float64)
    if window is not None:
        window_arr = signal.get_window(window, len(arr)) if isinstance(window, str) else np.asarray(window, dtype=np.float64)
        if len(window_arr) != len(arr):
            raise ValueError("Window length must match the signal length.")

    spec = scipy_fft.rfft(arr * window_arr, n=nfft)
    mags = np.abs(spec) / max(len(arr), 1)
    freqs = scipy_fft.rfftfreq(nfft, d=1 / fs)
    return freqs, 20 * np.log10(mags + EPSILON)


def spectrogram_power(
    sig: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a spectrogram in dB."""
    freqs, times, sxx = signal.spectrogram(
        np.asarray(sig, dtype=np.float64),
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    return freqs, times, 20 * np.log10(sxx + EPSILON)


def am_modulate(
    message: np.ndarray,
    carrier_freq: float,
    fs: float,
    mod_index: float = 0.8,
    carrier_amplitude: float = 1.0,
) -> np.ndarray:
    """AM modulate a message signal with a cosine carrier."""
    msg = np.asarray(message, dtype=np.float64)
    t = np.arange(len(msg), dtype=np.float64) / fs
    carrier = carrier_amplitude * np.cos(2 * np.pi * carrier_freq * t)
    return (1.0 + mod_index * msg) * carrier


def fm_modulate(
    message: np.ndarray,
    carrier_freq: float,
    fs: float,
    freq_dev: float = 2500.0,
    carrier_amplitude: float = 1.0,
) -> np.ndarray:
    """FM modulate a message signal."""
    msg = np.asarray(message, dtype=np.float64)
    t = np.arange(len(msg), dtype=np.float64) / fs
    phase_integral = np.cumsum(msg) / fs
    return carrier_amplitude * np.cos(
        (2 * np.pi * carrier_freq * t) + (2 * np.pi * freq_dev * phase_integral)
    )


def am_demodulate(
    rx_signal: np.ndarray,
    fs: float,
    audio_cutoff: float = 3500.0,
) -> np.ndarray:
    """AM demodulation via full-wave rectification and low-pass filtering."""
    rectified = np.abs(np.asarray(rx_signal, dtype=np.float64))
    sos_lp = butter(5, audio_cutoff, btype="low", fs=fs, output="sos")
    envelope = sosfilt(sos_lp, rectified)
    return normalize(envelope - np.mean(envelope))


def fm_demodulate(
    rx_signal: np.ndarray,
    fs: float,
    audio_cutoff: float = 3500.0,
) -> np.ndarray:
    """FM demodulation via analytic signal and phase differentiation."""
    analytic = hilbert(np.asarray(rx_signal, dtype=np.float64))
    inst_phase = np.unwrap(np.angle(analytic))
    demod = np.diff(inst_phase) * fs / (2 * np.pi)
    if demod.size == 0:
        return demod
    demod = np.append(demod, demod[-1])
    demod = demod - np.mean(demod)
    sos_lp = butter(5, audio_cutoff, btype="low", fs=fs, output="sos")
    return normalize(sosfilt(sos_lp, demod))


def add_awgn(
    signal_data: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add white Gaussian noise at a specified SNR in dB."""
    arr = np.asarray(signal_data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sig_power = np.mean(arr**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power), size=len(arr))
    return arr + noise, noise


def add_impulse_noise(
    signal_data: np.ndarray,
    fs: float,
    rate: float = 50.0,
    amplitude: float = 5.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Add short impulse spikes to simulate ignition or lightning noise."""
    arr = np.asarray(signal_data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = np.zeros_like(arr)
    num_spikes = int(rate * len(arr) / fs)
    if num_spikes <= 0:
        return arr.copy(), noise

    spike_locs = rng.integers(0, len(arr), size=num_spikes)
    for loc in spike_locs:
        width = int(rng.integers(3, 20))
        end = min(int(loc) + width, len(arr))
        polarity = rng.choice(np.array([-1.0, 1.0]))
        noise[int(loc):end] = polarity * amplitude * rng.uniform(0.5, 1.0)
    return arr + noise, noise


def pre_emphasis(audio: np.ndarray, fs: float, tau: float = 750e-6) -> np.ndarray:
    """Apply a first-order pre-emphasis filter."""
    alpha = np.exp(-1 / (fs * tau))
    return lfilter([1, -alpha], [1], np.asarray(audio, dtype=np.float64))


def de_emphasis(audio: np.ndarray, fs: float, tau: float = 750e-6) -> np.ndarray:
    """Apply the inverse of the pre-emphasis filter."""
    alpha = np.exp(-1 / (fs * tau))
    return lfilter([1], [1, -alpha], np.asarray(audio, dtype=np.float64))


def load_audio(
    filepath: str | os.PathLike[str],
    mono: bool = True,
    normalize_audio: bool = False,
    target_fs: float | None = None,
) -> tuple[int, np.ndarray]:
    """Load audio from WAV directly or convert other formats via ffmpeg."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".wav":
        rate, data = wavfile.read(path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(path),
                    "-sample_fmt",
                    "s16",
                    str(tmp_path),
                ],
                check=True,
                capture_output=True,
            )
            rate, data = wavfile.read(tmp_path)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to load non-WAV audio files.") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    arr = ensure_mono(data) if mono else np.asarray(data)
    arr = np.asarray(arr, dtype=np.float64)
    if normalize_audio:
        arr = normalize(arr)
    if target_fs is not None and target_fs != rate:
        arr = resample_signal(arr, rate, target_fs)
        rate = int(target_fs)
    return rate, arr


def fspl_db(d_km: float, f_mhz: float) -> float:
    """Compute free-space path loss in dB."""
    if d_km <= 0 or f_mhz <= 0:
        raise ValueError("Distance and frequency must be positive.")
    return float(20 * np.log10(d_km) + 20 * np.log10(f_mhz) + 32.44)


def load_complex_capture(
    filepath: str | os.PathLike[str],
    sample_rate: float | None = None,
) -> tuple[float | None, np.ndarray]:
    """Load a local IQ capture from .npy or .npz."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        iq = np.load(path)
        return sample_rate, np.asarray(iq, dtype=np.complex128)

    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "iq" in data:
            iq = data["iq"]
        elif "samples" in data:
            iq = data["samples"]
        else:
            raise ValueError("NPZ capture must contain 'iq' or 'samples'.")
        fs = sample_rate
        if fs is None and "sample_rate" in data:
            fs = float(np.asarray(data["sample_rate"]).reshape(-1)[0])
        return fs, np.asarray(iq, dtype=np.complex128)

    raise ValueError("Unsupported capture format. Use .npy or .npz for IQ captures.")


def complex_mix_down(iq: np.ndarray, fs: float, freq_shift: float) -> np.ndarray:
    """Frequency-shift a complex IQ signal by mixing with a complex exponential."""
    arr = np.asarray(iq, dtype=np.complex128)
    t = np.arange(len(arr), dtype=np.float64) / fs
    return arr * np.exp(-1j * 2 * np.pi * freq_shift * t)


def lowpass_filter(
    signal_data: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter to real or complex data."""
    arr = np.asarray(signal_data)
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    if np.iscomplexobj(arr):
        return sosfilt(sos, arr.real) + 1j * sosfilt(sos, arr.imag)
    return sosfilt(sos, arr.astype(np.float64))


def bandpass_filter(
    signal_data: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to real or complex data."""
    arr = np.asarray(signal_data)
    sos = butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")
    if np.iscomplexobj(arr):
        return sosfilt(sos, arr.real) + 1j * sosfilt(sos, arr.imag)
    return sosfilt(sos, arr.astype(np.float64))


def fm_demodulate_iq(
    iq_signal: np.ndarray,
    fs: float,
    audio_cutoff: float = 3500.0,
) -> np.ndarray:
    """FM demodulate a complex baseband IQ signal."""
    iq = np.asarray(iq_signal, dtype=np.complex128)
    if len(iq) < 2:
        return np.asarray(iq.real, dtype=np.float64)
    phase_step = np.angle(iq[1:] * np.conj(iq[:-1]))
    demod = np.concatenate([phase_step, phase_step[-1:]]) * fs / (2 * np.pi)
    demod = demod - np.mean(demod)
    return normalize(lowpass_filter(demod, cutoff_hz=audio_cutoff, fs=fs))


def am_demodulate_iq(
    iq_signal: np.ndarray,
    fs: float,
    audio_cutoff: float = 3500.0,
) -> np.ndarray:
    """AM demodulate a complex baseband IQ signal via envelope detection."""
    env = np.abs(np.asarray(iq_signal, dtype=np.complex128))
    env = env - np.mean(env)
    return normalize(lowpass_filter(env, cutoff_hz=audio_cutoff, fs=fs))


def synthesize_fm_iq(
    message: np.ndarray,
    fs: float,
    carrier_offset: float = 0.0,
    freq_dev: float = 2500.0,
) -> np.ndarray:
    """Generate synthetic FM IQ samples around a complex carrier offset."""
    msg = np.asarray(message, dtype=np.float64)
    t = np.arange(len(msg), dtype=np.float64) / fs
    phase = 2 * np.pi * carrier_offset * t + 2 * np.pi * freq_dev * np.cumsum(msg) / fs
    return np.exp(1j * phase)


def probe_rtlsdr() -> dict[str, object]:
    """Check whether pyrtlsdr is installed and whether a device can be opened."""
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        return {
            "installed": False,
            "available": False,
            "message": "pyrtlsdr is not installed.",
        }

    try:
        sdr = RtlSdr()
        sdr.close()
        return {
            "installed": True,
            "available": True,
            "message": "RTL-SDR device detected.",
        }
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {
            "installed": True,
            "available": False,
            "message": f"pyrtlsdr is installed, but no usable device was opened: {exc}",
        }


def _require_matplotlib_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _require_ipython_display():
    try:
        from IPython.display import Audio, display
    except ImportError as exc:
        raise RuntimeError("IPython is required for notebook audio helpers.") from exc
    return Audio, display


def _require_ipywidgets():
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError(
            "ipywidgets is required for widget helpers. Install it from requirements.txt."
        ) from exc
    return widgets


def plot_waveform(
    signal_data: np.ndarray,
    fs: float,
    ax=None,
    title: str | None = None,
    time_unit: str = "ms",
    color: str = "tab:blue",
):
    """Plot a signal in the time domain."""
    plt = _require_matplotlib_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    arr = np.asarray(signal_data, dtype=np.float64)
    scale = 1e3 if time_unit == "ms" else 1.0
    label = "Time (ms)" if time_unit == "ms" else "Time (s)"
    t = np.arange(len(arr), dtype=np.float64) / fs
    ax.plot(t * scale, arr, color=color)
    ax.set_xlabel(label)
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title)
    return ax


def plot_spectrum(
    signal_data: np.ndarray,
    fs: float,
    ax=None,
    title: str | None = None,
    nfft: int | None = None,
    window: str | Iterable[float] | None = None,
    color: str = "tab:blue",
):
    """Plot a single-sided magnitude spectrum."""
    plt = _require_matplotlib_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    freqs, mags_db = power_spectrum(signal_data, fs, nfft=nfft, window=window)
    ax.plot(freqs, mags_db, color=color)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if title:
        ax.set_title(title)
    return ax


def plot_spectrogram(
    signal_data: np.ndarray,
    fs: float,
    ax=None,
    title: str | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    cmap: str = "magma",
):
    """Plot a spectrogram in dB."""
    plt = _require_matplotlib_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    freqs, times, sxx_db = spectrogram_power(
        signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    mesh = ax.pcolormesh(times, freqs, sxx_db, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)
    return ax, mesh


def audio_player(signal_data: np.ndarray, rate: int, normalize_audio: bool = True):
    """Build an IPython audio player for notebook playback."""
    Audio, _ = _require_ipython_display()
    arr = normalize(signal_data) if normalize_audio else np.asarray(signal_data, dtype=np.float64)
    return Audio(arr, rate=rate)


def audio_output_widget():
    """Create an ipywidgets.Output for notebook audio playback."""
    widgets = _require_ipywidgets()
    return widgets.Output()


def refresh_audio_widget(output, signal_data: np.ndarray, rate: int, normalize_audio: bool = True):
    """Replace the contents of an output widget with a fresh audio player."""
    _, display = _require_ipython_display()
    with output:
        output.clear_output(wait=True)
        display(audio_player(signal_data, rate=rate, normalize_audio=normalize_audio))


def float_slider(
    *,
    min_value: float,
    max_value: float,
    step: float,
    value: float,
    description: str,
    readout_format: str = ".2f",
    continuous_update: bool = False,
):
    """Create a consistent FloatSlider for notebook demos."""
    widgets = _require_ipywidgets()
    return widgets.FloatSlider(
        min=min_value,
        max=max_value,
        step=step,
        value=value,
        description=description,
        readout_format=readout_format,
        continuous_update=continuous_update,
    )


def int_slider(
    *,
    min_value: int,
    max_value: int,
    step: int,
    value: int,
    description: str,
    continuous_update: bool = False,
):
    """Create a consistent IntSlider for notebook demos."""
    widgets = _require_ipywidgets()
    return widgets.IntSlider(
        min=min_value,
        max=max_value,
        step=step,
        value=value,
        description=description,
        continuous_update=continuous_update,
    )


def dropdown(*, options, value, description: str):
    """Create a consistent Dropdown for notebook demos."""
    widgets = _require_ipywidgets()
    return widgets.Dropdown(options=options, value=value, description=description)
