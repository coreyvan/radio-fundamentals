from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


COMMON_SETUP = """import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rf_utils import *
from IPython.display import Audio, Markdown, display
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

%matplotlib widget

plt.rcParams.update({
    "figure.figsize": (12, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
"""


VOICE_SETUP = """VOICE_FILE = ROOT / "assets" / "local" / "my_voice.m4a"
WORK_FS = 96_000
PLAY_FS = 44_100

if VOICE_FILE.exists():
    raw_fs, raw_audio = load_audio(VOICE_FILE, normalize_audio=True)
    raw_audio = normalize(ensure_mono(raw_audio))
    raw_audio = raw_audio[: int(raw_fs * 5)]
    voice_work = normalize(resample_signal(raw_audio, raw_fs, WORK_FS))
    voice_play = normalize(resample_signal(raw_audio, raw_fs, PLAY_FS))
    print(f"Loaded {VOICE_FILE.name} at {raw_fs} Hz")
else:
    t_fallback = np.arange(0, 3.0, 1 / WORK_FS)
    voice_work = normalize(
        0.7 * np.sin(2 * np.pi * 220 * t_fallback)
        + 0.4 * np.sin(2 * np.pi * 440 * t_fallback)
        + 0.2 * np.sin(2 * np.pi * 880 * t_fallback)
    )
    voice_play = normalize(resample_signal(voice_work, WORK_FS, PLAY_FS))
    print(f"No local voice recording found at {VOICE_FILE}. Using a synthetic fallback.")

t_work = np.arange(len(voice_work)) / WORK_FS
"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)


notebooks = {
    ROOT / "03-frequency-domain/01-fourier-transform.ipynb": [
        md(
            """# Fourier Transform

This notebook splits a signal into its sinusoidal building blocks. It lifts the "baseband audio" and spectrum work out of the legacy `dsp.ipynb` notebook into a focused module on time-domain intuition, spectral peaks, and synthesis."""
        ),
        code(COMMON_SETUP),
        md(
            """## A Voice-Like Signal in Time and Frequency

The Fourier transform answers a simple question: *which sinusoids are present, and how much of each is there?* We start with a synthetic three-tone signal so the time waveform looks busy but the spectrum stays easy to read."""
        ),
        code(
            """fs = 44_100
t = np.arange(0, 0.04, 1 / fs)
sig = (
    0.9 * np.cos(2 * np.pi * 220 * t)
    + 0.5 * np.cos(2 * np.pi * 440 * t)
    + 0.2 * np.cos(2 * np.pi * 880 * t + np.pi / 6)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
plot_waveform(sig[:1500], fs=fs, ax=axes[0], title="Time Domain")
plot_spectrum(sig, fs=fs, ax=axes[1], title="Magnitude Spectrum")
axes[1].set_xlim(0, 2000)
axes[1].set_ylim(-100, 5)
plt.tight_layout()

display(Audio(normalize(sig), rate=fs))
"""
        ),
        md(
            """## Interactive Spectrum Synthesizer

The legacy notebook already showed that a waveform gets visually complicated long before the spectrum does. This widget makes that connection explicit: drag the component amplitudes and watch narrow spikes rearrange the waveform."""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_synth(f1=220.0, a1=0.9, f2=440.0, a2=0.5, f3=880.0, a3=0.2):
    fs = 44_100
    t = np.arange(0, 0.03, 1 / fs)
    sig = (
        a1 * np.cos(2 * np.pi * f1 * t)
        + a2 * np.cos(2 * np.pi * f2 * t)
        + a3 * np.cos(2 * np.pi * f3 * t)
    )
    axes[0].clear()
    axes[1].clear()
    plot_waveform(sig[:1200], fs=fs, ax=axes[0], title="Waveform")
    plot_spectrum(sig, fs=fs, ax=axes[1], title="Spectrum")
    axes[1].set_xlim(0, 3000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, sig, rate=fs)

controls = widgets.interactive(
    update_synth,
    f1=float_slider(min_value=100, max_value=1200, step=20, value=220, description="f1"),
    a1=float_slider(min_value=0, max_value=1, step=0.05, value=0.9, description="a1"),
    f2=float_slider(min_value=100, max_value=2000, step=20, value=440, description="f2"),
    a2=float_slider(min_value=0, max_value=1, step=0.05, value=0.5, description="a2"),
    f3=float_slider(min_value=100, max_value=3000, step=20, value=880, description="f3"),
    a3=float_slider(min_value=0, max_value=1, step=0.05, value=0.2, description="a3"),
)

display(controls, audio_out)
"""
        ),
        md(
            """## Beat Frequencies

Two nearby tones create a slow amplitude wobble called a beat. In the spectrum, nothing mysterious happens: you just see two close lines. In the time domain, their interference makes the envelope pulse."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_beats(base_freq=440.0, separation=4.0):
    fs = 44_100
    t = np.arange(0, 1.0, 1 / fs)
    sig = np.cos(2 * np.pi * base_freq * t) + np.cos(2 * np.pi * (base_freq + separation) * t)
    axes[0].clear()
    axes[1].clear()
    plot_waveform(sig[:6000], fs=fs, ax=axes[0], title="Beat Pattern")
    plot_spectrum(sig, fs=fs, ax=axes[1], title="Two Nearby Spectral Lines")
    axes[1].set_xlim(base_freq - 30, base_freq + separation + 30)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, sig, rate=fs)

controls = widgets.interactive(
    update_beats,
    base_freq=float_slider(min_value=200, max_value=1200, step=20, value=440, description="Base Hz"),
    separation=float_slider(min_value=0.5, max_value=20, step=0.5, value=4, description="Delta Hz"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## What to Try

- Increase `a3` until the spectrum shows a strong third harmonic and notice how much rougher the waveform sounds.
- Move `f2` close to `f1` and listen for beats before you can clearly hear the two tones separately.
- Set one amplitude to zero and confirm that one spectral spike disappears completely.

## Key Takeaway

The time waveform can look complicated while the spectrum stays simple. That is the core value of the Fourier transform: it gives you a clearer coordinate system for understanding signals."""
        ),
    ],
    ROOT / "03-frequency-domain/02-fft-in-practice.ipynb": [
        md(
            """# FFT in Practice

This notebook extracts the practical FFT material from the legacy notebooks: windowing, frequency resolution, and spectrograms. The goal is to show why FFT settings change what you think you are seeing."""
        ),
        code(COMMON_SETUP),
        md(
            """## Resolution and Leakage

If a tone does not fit perfectly into the observation window, its energy leaks into neighboring FFT bins. Window functions trade peak sharpness for lower sidelobes."""
        ),
        code(
            """fs = 48_000
duration = 0.03
t = np.arange(0, duration, 1 / fs)
tone = np.cos(2 * np.pi * 1037 * t)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
plot_waveform(tone[:1200], fs=fs, ax=axes[0], title="Off-Bin Tone")

for name in ["boxcar", "hann", "hamming", "blackman"]:
    freqs, mags_db = power_spectrum(tone, fs=fs, nfft=8192, window=name)
    axes[1].plot(freqs, mags_db, label=name)

axes[1].set_xlim(700, 1400)
axes[1].set_ylim(-140, 5)
axes[1].set_title("Window Comparison")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude (dB)")
axes[1].legend()
plt.tight_layout()
"""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 3.5))

def update_window(window="hann", nfft=4096):
    ax.clear()
    freqs, mags_db = power_spectrum(tone, fs=fs, nfft=nfft, window=window)
    ax.plot(freqs, mags_db, color="tab:orange")
    ax.set_xlim(700, 1400)
    ax.set_ylim(-140, 5)
    ax.set_title(f"{window} window, nfft={nfft}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_window,
    window=dropdown(options=["boxcar", "hann", "hamming", "blackman"], value="hann", description="Window"),
    nfft=int_slider(min_value=1024, max_value=16384, step=1024, value=4096, description="NFFT"),
)
display(controls)
"""
        ),
        md(
            """## Spectrograms

The spectrogram section in `dsp.ipynb` is the short-time Fourier transform in action. Each vertical slice is an FFT over a short window; stacked together, those FFTs show how frequency content evolves over time."""
        ),
        code(
            """fs = 48_000
t = np.arange(0, 1.5, 1 / fs)
chirp = signal.chirp(t, f0=300, f1=6000, t1=t[-1], method="linear")
am_chirp = am_modulate(chirp, carrier_freq=10_000, fs=fs, mod_index=0.7)
fm_chirp = fm_modulate(chirp, carrier_freq=10_000, fs=fs, freq_dev=2500)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_spectrogram(am_chirp, fs=fs, ax=axes[0], title="AM Chirp Spectrogram")
plot_spectrogram(fm_chirp, fs=fs, ax=axes[1], title="FM Chirp Spectrogram")
for ax in axes:
    ax.set_ylim(0, 15_000)
plt.tight_layout()
"""
        ),
        md(
            """## What to Try

- Compare `boxcar` and `blackman` windows and look at how the sidelobe floor changes.
- Double `nfft` and notice that the line positions become smoother, but the underlying data did not gain new information.
- Change the chirp start and end frequencies and watch the spectrogram slope change.

## Key Takeaway

The FFT is not just "run it and trust it." Window choice, FFT length, and segment size all change the picture you get, and radio tools depend on choosing those parameters deliberately."""
        ),
    ],
    ROOT / "04-filters/01-filter-concepts.ipynb": [
        md(
            """# Filter Concepts

This notebook spins the voice-band filtering material out of `am_vs_fm_audio_demo.ipynb` into a standalone lesson on low-pass, high-pass, and band-pass behavior."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Why Radios Filter Audio

Most narrowband voice links only need about 300 Hz to 3 kHz of audio. Everything below that wastes deviation and everything above that mostly carries hiss or sharp transients."""
        ),
        code(
            """voice_bp = signal.sosfilt(
    signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"),
    voice_work,
)
voice_bp = normalize(voice_bp)

fig, axes = plt.subplots(2, 2, figsize=(13, 6))
plot_waveform(voice_work[:10_000], fs=WORK_FS, ax=axes[0, 0], title="Original Voice")
plot_waveform(voice_bp[:10_000], fs=WORK_FS, ax=axes[0, 1], title="Band-Limited Voice")
plot_spectrum(voice_work, fs=WORK_FS, ax=axes[1, 0], title="Original Spectrum")
plot_spectrum(voice_bp, fs=WORK_FS, ax=axes[1, 1], title="Band-Limited Spectrum")
axes[1, 0].set_xlim(0, 8000)
axes[1, 1].set_xlim(0, 8000)
axes[1, 0].set_ylim(-100, 5)
axes[1, 1].set_ylim(-100, 5)
plt.tight_layout()

display(Markdown("**Original audio**"))
display(audio_player(resample_signal(voice_work, WORK_FS, PLAY_FS), rate=PLAY_FS))
display(Markdown("**Band-limited audio**"))
display(audio_player(resample_signal(voice_bp, WORK_FS, PLAY_FS), rate=PLAY_FS))
"""
        ),
        md(
            """## Interactive Low-Pass Filtering

Dragging the cutoff down is the quickest way to hear what filters do. As the cutoff falls, intelligibility gives way to muffled energy and then mostly pitch contour."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_filter(cutoff=2500.0):
    sos = signal.butter(5, cutoff, btype="low", fs=WORK_FS, output="sos")
    filtered = normalize(signal.sosfilt(sos, voice_work))
    axes[0].clear()
    axes[1].clear()
    plot_waveform(filtered[:10_000], fs=WORK_FS, ax=axes[0], title=f"Low-pass, cutoff={cutoff:.0f} Hz")
    plot_spectrum(filtered, fs=WORK_FS, ax=axes[1], title="Filtered Spectrum")
    axes[1].set_xlim(0, 8000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(filtered, WORK_FS, PLAY_FS), rate=PLAY_FS)

controls = widgets.interactive(
    update_filter,
    cutoff=float_slider(min_value=300, max_value=5000, step=100, value=2500, description="Cutoff"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Filters are deliberate frequency selectors. The radio "sound" you hear is not just the microphone or speaker; it is shaped by bandwidth limits all along the chain."""
        ),
    ],
    ROOT / "04-filters/02-filter-design.ipynb": [
        md(
            """# Filter Design

This notebook extracts the receive-chain bandpass design material from `dsp.ipynb`. It focuses on cutoff choice, order, and how the transfer function reshapes both signal and audio."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Order Changes the Transition Band

Higher-order filters roll off faster, but they also become more selective and can add more ringing or delay. The legacy notebook compared Butterworth orders directly; this notebook makes that comparison reusable."""
        ),
        code(
            """orders = [2, 4, 8]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

for order in orders:
    sos = signal.butter(order, [300, 3000], btype="band", fs=WORK_FS, output="sos")
    w, h = signal.sosfreqz(sos, worN=8192, fs=WORK_FS)
    axes[0].plot(w, 20 * np.log10(np.abs(h) + 1e-12), label=f"order {order}")
    axes[1].plot(w, np.unwrap(np.angle(h)), label=f"order {order}")

axes[0].set_xlim(0, 8000)
axes[0].set_ylim(-80, 5)
axes[0].set_title("Magnitude Response")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Magnitude (dB)")
axes[0].legend()
axes[1].set_xlim(0, 8000)
axes[1].set_title("Phase Response")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Radians")
axes[1].legend()
plt.tight_layout()
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_design(order=4, low_cut=300.0, high_cut=3000.0):
    axes[0].clear()
    axes[1].clear()
    sos = signal.butter(order, [low_cut, high_cut], btype="band", fs=WORK_FS, output="sos")
    w, h = signal.sosfreqz(sos, worN=8192, fs=WORK_FS)
    filtered = normalize(signal.sosfilt(sos, voice_work))
    axes[0].plot(w, 20 * np.log10(np.abs(h) + 1e-12), color="tab:green")
    axes[0].set_xlim(0, 8000)
    axes[0].set_ylim(-80, 5)
    axes[0].set_title("Designed Filter Response")
    plot_spectrum(filtered, fs=WORK_FS, ax=axes[1], title="Filtered Voice Spectrum", color="tab:red")
    axes[1].set_xlim(0, 8000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(filtered, WORK_FS, PLAY_FS), rate=PLAY_FS)

controls = widgets.interactive(
    update_design,
    order=int_slider(min_value=2, max_value=10, step=2, value=4, description="Order"),
    low_cut=float_slider(min_value=100, max_value=1000, step=50, value=300, description="Low Hz"),
    high_cut=float_slider(min_value=1500, max_value=5000, step=100, value=3000, description="High Hz"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Filter design is a tradeoff problem, not a single "best" choice. Order and cutoff control selectivity, delay, and how natural the recovered audio sounds."""
        ),
    ],
    ROOT / "05-am-modulation/01-am-fundamentals.ipynb": [
        md(
            """# AM Fundamentals

This notebook breaks the AM section out of both legacy notebooks. It keeps the core ideas together: envelope shape, sidebands, modulation index, and envelope detection."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Envelope Modulation

AM multiplies a carrier by a shifted version of the message:

$$s_{AM}(t) = [1 + m x(t)] \\cos(2\\pi f_c t)$$

The envelope follows the message as long as the modulation index stays below 1."""
        ),
        code(
            """carrier_freq = 20_000
message = signal.sosfilt(signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"), voice_work)
message = normalize(message)
am_signal = am_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, mod_index=0.8)
am_demod = am_demodulate(am_signal, fs=WORK_FS)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
win = slice(0, 8000)
axes[0].plot(t_work[win] * 1000, message[win], color="tab:green")
axes[0].set_title("Message")
axes[0].set_xlabel("Time (ms)")
axes[1].plot(t_work[win] * 1000, am_signal[win], color="tab:blue", linewidth=0.6)
axes[1].set_title("AM Waveform")
axes[1].set_xlabel("Time (ms)")
plot_spectrum(am_signal, fs=WORK_FS, ax=axes[2], title="AM Spectrum")
axes[2].set_xlim(0, 30_000)
axes[2].set_ylim(-100, 5)
plt.tight_layout()

display(Markdown("**Demodulated AM audio**"))
display(audio_player(resample_signal(am_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_am(mod_index=0.8):
    am_signal = am_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, mod_index=mod_index)
    demod = am_demodulate(am_signal, fs=WORK_FS)
    axes[0].clear()
    axes[1].clear()
    win = slice(0, 6000)
    axes[0].plot(t_work[win] * 1000, am_signal[win], color="tab:blue", linewidth=0.6)
    axes[0].set_title(f"AM Waveform, m={mod_index:.2f}")
    axes[0].set_xlabel("Time (ms)")
    plot_spectrum(am_signal, fs=WORK_FS, ax=axes[1], title="AM Spectrum")
    axes[1].set_xlim(0, 30_000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(demod, WORK_FS, PLAY_FS), rate=PLAY_FS)

controls = widgets.interactive(
    update_am,
    mod_index=float_slider(min_value=0, max_value=1.5, step=0.05, value=0.8, description="m"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## What to Try

- Push `m` above 1 and listen for overmodulation distortion.
- Watch the sidebands stay symmetric around the carrier.
- Compare the message waveform to the AM envelope over a short zoomed-in segment.

## Key Takeaway

AM is visually intuitive because the message rides in the envelope, but that same amplitude dependence makes it more vulnerable to noise and impulsive interference."""
        ),
    ],
    ROOT / "06-fm-modulation/01-fm-fundamentals.ipynb": [
        md(
            """# FM Fundamentals

This notebook pulls together the core FM material from the legacy notebooks: constant-envelope modulation, deviation, sidebands, and phase-derivative demodulation."""
        ),
        code(COMMON_SETUP + "\nfrom scipy.special import jv\n"),
        code(VOICE_SETUP),
        md(
            """## FM Moves Information into Instantaneous Frequency

$$s_{FM}(t) = \\cos\\left(2\\pi f_c t + 2\\pi k_f \\int_0^t m(\\tau) d\\tau\\right)$$

The envelope stays constant. What changes is the density of zero crossings and the instantaneous phase slope."""
        ),
        code(
            """carrier_freq = 20_000
message = signal.sosfilt(signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"), voice_work)
message = normalize(message)
freq_dev = 2_500
fm_signal = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=freq_dev)
fm_demod = fm_demodulate(fm_signal, fs=WORK_FS)

analytic = signal.hilbert(fm_signal)
inst_freq = np.diff(np.unwrap(np.angle(analytic))) * WORK_FS / (2 * np.pi)
inst_freq = np.append(inst_freq, inst_freq[-1])

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
win = slice(0, 8000)
axes[0].plot(t_work[win] * 1000, fm_signal[win], color="tab:orange", linewidth=0.6)
axes[0].set_title("FM Waveform")
axes[0].set_xlabel("Time (ms)")
axes[1].plot(t_work[win] * 1000, inst_freq[win] - np.mean(inst_freq), color="tab:red", linewidth=0.8)
axes[1].set_title("Instantaneous Frequency")
axes[1].set_xlabel("Time (ms)")
plot_spectrum(fm_signal, fs=WORK_FS, ax=axes[2], title="FM Spectrum")
axes[2].set_xlim(0, 30_000)
axes[2].set_ylim(-100, 5)
plt.tight_layout()

display(Markdown("**FM demodulated audio**"))
display(audio_player(resample_signal(fm_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
"""
        ),
        code(
            """betas = np.linspace(0, 5, 300)
orders = range(5)
fig, ax = plt.subplots(figsize=(10, 3.5))
for order in orders:
    ax.plot(betas, jv(order, betas), label=f"J{order}")
ax.set_title("Bessel Functions and FM Sideband Strength")
ax.set_xlabel("Modulation index beta")
ax.set_ylabel("Amplitude")
ax.legend(ncol=5)
plt.tight_layout()
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_fm(freq_dev=2500.0):
    fm_signal = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=freq_dev)
    demod = fm_demodulate(fm_signal, fs=WORK_FS)
    axes[0].clear()
    axes[1].clear()
    plot_waveform(fm_signal[:8000], fs=WORK_FS, ax=axes[0], title=f"FM Waveform, dev={freq_dev:.0f} Hz")
    plot_spectrum(fm_signal, fs=WORK_FS, ax=axes[1], title="FM Spectrum")
    axes[1].set_xlim(0, 30_000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(demod, WORK_FS, PLAY_FS), rate=PLAY_FS)

controls = widgets.interactive(
    update_fm,
    freq_dev=float_slider(min_value=500, max_value=6000, step=100, value=2500, description="Dev Hz"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

FM hides the message in phase and frequency rather than amplitude. That makes it harder to understand at first glance, but it is also why FM handles amplitude noise so much better."""
        ),
    ],
    ROOT / "06-fm-modulation/02-narrowband-vs-wideband.ipynb": [
        md(
            """# Narrowband vs Wideband FM

This notebook isolates the GMRS-style deviation comparison from the legacy audio demo. It compares narrowband (±2.5 kHz) and wideband (±5 kHz) FM in both spectrum and recovered audio."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Same Message, Different Deviation

Wider deviation improves recovered audio quality and noise performance, but it occupies more spectrum. That is the central tradeoff between narrowband and wideband FM."""
        ),
        code(
            """message = signal.sosfilt(signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"), voice_work)
message = normalize(message)
carrier_freq = 20_000

fm_narrow = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=2_500)
fm_wide = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=5_000)
demod_narrow = fm_demodulate(fm_narrow, fs=WORK_FS)
demod_wide = fm_demodulate(fm_wide, fs=WORK_FS)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
plot_spectrum(fm_narrow, fs=WORK_FS, ax=axes[0], title="Narrowband FM Spectrum", color="tab:orange")
plot_spectrum(fm_wide, fs=WORK_FS, ax=axes[1], title="Wideband FM Spectrum", color="tab:red")
for ax in axes:
    ax.set_xlim(0, 35_000)
    ax.set_ylim(-100, 5)
plt.tight_layout()

display(Markdown("**Narrowband FM demodulated audio**"))
display(audio_player(resample_signal(demod_narrow, WORK_FS, PLAY_FS), rate=PLAY_FS))
display(Markdown("**Wideband FM demodulated audio**"))
display(audio_player(resample_signal(demod_wide, WORK_FS, PLAY_FS), rate=PLAY_FS))
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_deviation(freq_dev=2500.0):
    fm_sig = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=freq_dev)
    demod = fm_demodulate(fm_sig, fs=WORK_FS)
    axes[0].clear()
    axes[1].clear()
    plot_waveform(fm_sig[:8000], fs=WORK_FS, ax=axes[0], title=f"Waveform, dev={freq_dev:.0f} Hz")
    plot_spectrum(fm_sig, fs=WORK_FS, ax=axes[1], title="Spectrum")
    axes[1].set_xlim(0, 35_000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(demod, WORK_FS, PLAY_FS), rate=PLAY_FS)

controls = widgets.interactive(
    update_deviation,
    freq_dev=float_slider(min_value=1500, max_value=6000, step=100, value=2500, description="Dev Hz"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Deviation is a spectrum budget decision. More deviation generally buys better audio and better FM noise performance, but it costs occupied bandwidth."""
        ),
    ],
    ROOT / "06-fm-modulation/03-pre-emphasis-de-emphasis.ipynb": [
        md(
            """# Pre-Emphasis and De-Emphasis

This notebook extracts the pre/de-emphasis comparison from the legacy audio demo. It shows how FM systems tilt the audio spectrum before transmission and undo that tilt after demodulation to suppress high-frequency hiss."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Spectral Tilt as a Noise Countermeasure

FM demodulated noise rises with frequency. Pre-emphasis boosts high audio frequencies before transmission, and de-emphasis cuts them back down after reception, which also knocks down the extra hiss."""
        ),
        code(
            """message = signal.sosfilt(signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"), voice_work)
message = normalize(message)
carrier_freq = 20_000
freq_dev = 2_500

pre = normalize(pre_emphasis(message, WORK_FS))
fm_plain = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=freq_dev)
fm_emph = fm_modulate(pre, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=freq_dev)

fm_plain_noisy, _ = add_awgn(fm_plain, 10, seed=42)
fm_emph_noisy, _ = add_awgn(fm_emph, 10, seed=42)

plain_demod = fm_demodulate(fm_plain_noisy, fs=WORK_FS)
emph_demod = normalize(de_emphasis(fm_demodulate(fm_emph_noisy, fs=WORK_FS), WORK_FS))

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
plot_spectrum(message, fs=WORK_FS, ax=axes[0], title="Original Audio Spectrum")
plot_spectrum(pre, fs=WORK_FS, ax=axes[1], title="Pre-Emphasized Spectrum")
axes[0].set_xlim(0, 6000)
axes[1].set_xlim(0, 6000)
plot_spectrum(emph_demod, fs=WORK_FS, ax=axes[2], title="Recovered Audio After De-Emphasis")
axes[2].set_xlim(0, 6000)
for ax in axes:
    ax.set_ylim(-100, 5)
plt.tight_layout()

display(Markdown("**FM without pre/de-emphasis**"))
display(audio_player(resample_signal(plain_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
display(Markdown("**FM with pre/de-emphasis**"))
display(audio_player(resample_signal(emph_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_emphasis(snr_db=10.0):
    fm_plain_noisy, _ = add_awgn(fm_plain, snr_db, seed=42)
    fm_emph_noisy, _ = add_awgn(fm_emph, snr_db, seed=42)
    plain_demod = fm_demodulate(fm_plain_noisy, fs=WORK_FS)
    emph_demod = normalize(de_emphasis(fm_demodulate(fm_emph_noisy, fs=WORK_FS), WORK_FS))
    axes[0].clear()
    axes[1].clear()
    plot_spectrum(plain_demod, fs=WORK_FS, ax=axes[0], title="No Emphasis")
    plot_spectrum(emph_demod, fs=WORK_FS, ax=axes[1], title="With Pre/De-Emphasis")
    for ax in axes:
        ax.set_xlim(0, 6000)
        ax.set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(emph_demod, WORK_FS, PLAY_FS), rate=PLAY_FS)

controls = widgets.interactive(
    update_emphasis,
    snr_db=float_slider(min_value=0, max_value=30, step=1, value=10, description="SNR dB"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Pre/de-emphasis does not change the message content. It changes where the system spends its SNR budget, which is why it becomes more audible as the channel gets noisy."""
        ),
    ],
    ROOT / "10-noise-and-sensitivity/01-thermal-noise.ipynb": [
        md(
            """# Thermal Noise

This notebook promotes the noise-floor and sensitivity ideas that were implicit in the legacy AM/FM demo into a dedicated lesson. It connects the `kTB` equation to practical receiver bandwidth and noise figure."""
        ),
        code(COMMON_SETUP),
        md(
            """## Room-Temperature Noise Power

At 290 K, thermal noise density is approximately `-174 dBm/Hz`. Over a receiver bandwidth `B`, the noise floor is:

$$P_n = -174\\,\\text{dBm/Hz} + 10\\log_{10}(B) + NF$$

where `NF` is the receiver noise figure in dB."""
        ),
        code(
            """def thermal_noise_dbm(bandwidth_hz, noise_figure_db=0.0):
    return -174 + 10 * np.log10(bandwidth_hz) + noise_figure_db

bandwidths = np.array([2500, 6250, 12_500, 25_000, 200_000])
for bw in bandwidths:
    print(f"{bw:>7,d} Hz -> {thermal_noise_dbm(bw, noise_figure_db=6):6.2f} dBm with 6 dB NF")
"""
        ),
        code(
            """out = widgets.Output()

def update_noise_floor(bandwidth_hz=12_500.0, noise_figure_db=6.0, required_sinad=12.0):
    with out:
        out.clear_output(wait=True)
        noise_floor = thermal_noise_dbm(bandwidth_hz, noise_figure_db)
        sensitivity = noise_floor + required_sinad
        print(f"Noise floor: {noise_floor:.2f} dBm")
        print(f"Estimated sensitivity for {required_sinad:.1f} dB SINAD: {sensitivity:.2f} dBm")

controls = widgets.interactive(
    update_noise_floor,
    bandwidth_hz=float_slider(min_value=2500, max_value=200000, step=2500, value=12500, description="BW Hz", readout_format=".0f"),
    noise_figure_db=float_slider(min_value=0, max_value=15, step=0.5, value=6, description="NF dB"),
    required_sinad=float_slider(min_value=3, max_value=20, step=0.5, value=12, description="SINAD"),
)
display(controls, out)
"""
        ),
        md(
            """## From Equations to Samples

We can simulate the effect of thermal-like white noise by adding AWGN to a tone and watching the spectrum floor rise as SNR drops."""
        ),
        code(
            """fs = 48_000
t, tone = generate_tone(freq=1000, duration=0.2, fs=fs, amplitude=1.0)

fig, ax = plt.subplots(figsize=(10, 3.5))

def update_awgn(snr_db=20.0):
    ax.clear()
    noisy, _ = add_awgn(tone, snr_db=snr_db, seed=42)
    plot_spectrum(noisy, fs=fs, ax=ax, title=f"Tone + AWGN, SNR={snr_db:.0f} dB")
    ax.set_xlim(0, 4000)
    ax.set_ylim(-120, 5)
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_awgn,
    snr_db=float_slider(min_value=0, max_value=40, step=1, value=20, description="SNR dB"),
)
display(controls)
"""
        ),
        md(
            """## Key Takeaway

Thermal noise is not a special pathology; it is the baseline. Receiver sensitivity starts with `kTB`, then every extra bandwidth and every dB of noise figure makes the floor worse."""
        ),
    ],
    ROOT / "10-noise-and-sensitivity/02-snr-and-intelligibility.ipynb": [
        md(
            """# SNR and Intelligibility

This notebook extracts the centerpiece of the legacy audio demo: the point where AM and FM experience the same noise and sound very different. It also keeps the impulse-noise comparison because intelligibility is about what your ears can still decode, not just a plotted SNR."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Same Noise, Different Modulation

This is the main A/B comparison from the old notebook. We band-limit the message, modulate it both ways, inject the same noise realization, then compare the recovered audio."""
        ),
        code(
            """message = signal.sosfilt(signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"), voice_work)
message = normalize(message)
carrier_freq = 20_000
am_signal = am_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, mod_index=0.8)
fm_signal = fm_modulate(message, carrier_freq=carrier_freq, fs=WORK_FS, freq_dev=2_500)
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(2, 2, figsize=(13, 7))

def update_snr(snr_db=20.0):
    am_noisy, _ = add_awgn(am_signal, snr_db=snr_db, seed=42)
    fm_noisy, _ = add_awgn(fm_signal, snr_db=snr_db, seed=42)
    am_demod = am_demodulate(am_noisy, fs=WORK_FS)
    fm_demod = fm_demodulate(fm_noisy, fs=WORK_FS)

    for ax in axes.flat:
        ax.clear()

    plot_waveform(am_noisy[:6000], fs=WORK_FS, ax=axes[0, 0], title="AM + Noise")
    plot_waveform(fm_noisy[:6000], fs=WORK_FS, ax=axes[0, 1], title="FM + Noise", color="tab:orange")
    plot_spectrum(am_demod, fs=WORK_FS, ax=axes[1, 0], title="AM Demodulated Spectrum")
    plot_spectrum(fm_demod, fs=WORK_FS, ax=axes[1, 1], title="FM Demodulated Spectrum", color="tab:orange")
    for ax in axes[1]:
        ax.set_xlim(0, 6000)
        ax.set_ylim(-100, 5)
    fig.canvas.draw_idle()

    with audio_out:
        audio_out.clear_output(wait=True)
        display(Markdown(f"**AM demodulated, SNR={snr_db:.0f} dB**"))
        display(audio_player(resample_signal(am_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
        display(Markdown(f"**FM demodulated, SNR={snr_db:.0f} dB**"))
        display(audio_player(resample_signal(fm_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))

controls = widgets.interactive(
    update_snr,
    snr_db=float_slider(min_value=0, max_value=30, step=1, value=20, description="SNR dB"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Impulse Noise

Impulse noise is a realistic stress test for land-mobile audio. AM treats those spikes as part of the amplitude envelope; FM largely rejects them unless they are strong enough to disrupt phase tracking."""
        ),
        code(
            """am_impulse, _ = add_impulse_noise(am_signal, fs=WORK_FS, rate=80, amplitude=4.0, seed=7)
fm_impulse, _ = add_impulse_noise(fm_signal, fs=WORK_FS, rate=80, amplitude=4.0, seed=7)
am_impulse_demod = am_demodulate(am_impulse, fs=WORK_FS)
fm_impulse_demod = fm_demodulate(fm_impulse, fs=WORK_FS)

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
plot_waveform(am_impulse[:10_000], fs=WORK_FS, ax=axes[0], title="AM + Impulse Noise")
plot_waveform(fm_impulse[:10_000], fs=WORK_FS, ax=axes[1], title="FM + Impulse Noise", color="tab:orange")
plt.tight_layout()

display(Markdown("**AM with impulse noise**"))
display(audio_player(resample_signal(am_impulse_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
display(Markdown("**FM with impulse noise**"))
display(audio_player(resample_signal(fm_impulse_demod, WORK_FS, PLAY_FS), rate=PLAY_FS))
"""
        ),
        md(
            """## What to Try

- Step SNR from 30 dB down to 0 dB and listen for when AM becomes tiring before it becomes fully unintelligible.
- Compare the spectral tilt of the demodulated FM noise to the flatter AM case.
- Listen to the impulse-noise examples and decide which artifacts are easier for your ear to ignore.

## Key Takeaway

Intelligibility is the human consequence of SNR. The reason FM sounds more robust is not marketing language; it is a direct result of how the modulation stores information."""
        ),
    ],
}


for path, cells in notebooks.items():
    write_notebook(path, cells)
    print(f"Wrote {path.relative_to(ROOT)}")
