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
else:
    t_fallback = np.arange(0, 3.0, 1 / WORK_FS)
    voice_work = normalize(
        0.7 * np.sin(2 * np.pi * 220 * t_fallback)
        + 0.4 * np.sin(2 * np.pi * 440 * t_fallback)
        + 0.2 * np.sin(2 * np.pi * 880 * t_fallback)
    )
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


NOTEBOOKS = {
    ROOT / "01-signals-and-waves/01-what-is-a-signal.ipynb": [
        md(
            """# What Is a Signal?

This notebook starts the course at the physical and mathematical beginning: a signal is just a quantity that changes over time. We use sine waves to build intuition for amplitude, frequency, and phase because they are the atoms of DSP."""
        ),
        code(COMMON_SETUP),
        md(
            """## One Tone, Three Knobs

For a cosine,

$$s(t) = A \\cos(2 \\pi f t + \\phi)$$

amplitude changes height, frequency changes how fast it oscillates, and phase shifts where the cycle starts."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_tone(freq=440.0, amplitude=0.8, phase=0.0):
    t, sig = generate_tone(freq=freq, duration=0.02, fs=44_100, amplitude=amplitude, phase=phase)
    axes[0].clear()
    axes[1].clear()
    plot_waveform(sig, fs=44_100, ax=axes[0], title="Waveform")
    plot_spectrum(sig, fs=44_100, ax=axes[1], title="Spectrum")
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, sig, rate=44_100)

controls = widgets.interactive(
    update_tone,
    freq=float_slider(min_value=20, max_value=4000, step=10, value=440, description="Freq Hz"),
    amplitude=float_slider(min_value=0, max_value=1, step=0.05, value=0.8, description="Amp"),
    phase=float_slider(min_value=0, max_value=2 * np.pi, step=0.1, value=0.0, description="Phase"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Phase Only Matters Relative to Something

A single tone shifted in phase sounds the same to your ear, but phase becomes important when two signals interact. Overlaying two equal tones makes the shift visible immediately."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 3.5))

def update_phase_difference(phase=0.0):
    ax.clear()
    t, sig1 = generate_tone(freq=300, duration=0.02, fs=44_100, amplitude=1.0, phase=0.0)
    _, sig2 = generate_tone(freq=300, duration=0.02, fs=44_100, amplitude=1.0, phase=phase)
    ax.plot(t * 1e3, sig1, label="Reference")
    ax.plot(t * 1e3, sig2, label="Shifted", alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Phase offset = {phase:.2f} rad")
    ax.legend()
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_phase_difference,
    phase=float_slider(min_value=0, max_value=2 * np.pi, step=0.1, value=0.0, description="Phase"),
)
display(controls)
"""
        ),
        md(
            """## What to Try

- Double the frequency and notice that the pitch rises while the waveform compresses in time.
- Set amplitude near zero and confirm that the spectrum peak drops with it.
- Sweep phase through `pi` while comparing two overlaid tones and watch the shift without hearing a pitch change.

## Key Takeaway

A signal is just a time-varying quantity, but sinusoids are the special case that make every later DSP concept manageable. If you can reason about amplitude, frequency, and phase, the rest of the course has a base to stand on."""
        ),
    ],
    ROOT / "01-signals-and-waves/02-adding-signals.ipynb": [
        md(
            """# Adding Signals

This notebook covers superposition: signals add sample by sample. That simple rule explains chords, harmonics, constructive and destructive interference, and beat frequencies."""
        ),
        code(COMMON_SETUP),
        md(
            """## Superposition

If two sources are present at once, the receiver sees their sum. There is no special "mixing" required for simple addition in the time domain."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_sum(f1=440.0, a1=0.8, f2=660.0, a2=0.6):
    fs = 44_100
    t = np.arange(0, 0.03, 1 / fs)
    sig1 = a1 * np.cos(2 * np.pi * f1 * t)
    sig2 = a2 * np.cos(2 * np.pi * f2 * t)
    total = sig1 + sig2
    axes[0].clear()
    axes[1].clear()
    plot_waveform(total[:1500], fs=fs, ax=axes[0], title="Sum in Time")
    plot_spectrum(total, fs=fs, ax=axes[1], title="Sum in Frequency")
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, normalize(total), rate=fs)

controls = widgets.interactive(
    update_sum,
    f1=float_slider(min_value=100, max_value=1200, step=10, value=440, description="f1"),
    a1=float_slider(min_value=0, max_value=1, step=0.05, value=0.8, description="a1"),
    f2=float_slider(min_value=100, max_value=1600, step=10, value=660, description="f2"),
    a2=float_slider(min_value=0, max_value=1, step=0.05, value=0.6, description="a2"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Build a Chord

Harmonics and chords are just sums of tones. The waveform gets more complex, but the spectrum simply lists the frequencies that are present."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_chord(root=220.0, third_gain=0.7, fifth_gain=0.6, octave_gain=0.4):
    fs = 44_100
    t = np.arange(0, 0.5, 1 / fs)
    freqs = [root, root * 5 / 4, root * 3 / 2, root * 2]
    gains = [1.0, third_gain, fifth_gain, octave_gain]
    chord = sum(g * np.cos(2 * np.pi * f * t) for f, g in zip(freqs, gains))
    axes[0].clear()
    axes[1].clear()
    plot_waveform(chord[:2500], fs=fs, ax=axes[0], title="Chord Waveform")
    plot_spectrum(chord, fs=fs, ax=axes[1], title="Chord Spectrum")
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, normalize(chord), rate=fs)

controls = widgets.interactive(
    update_chord,
    root=float_slider(min_value=100, max_value=400, step=5, value=220, description="Root"),
    third_gain=float_slider(min_value=0, max_value=1, step=0.05, value=0.7, description="3rd"),
    fifth_gain=float_slider(min_value=0, max_value=1, step=0.05, value=0.6, description="5th"),
    octave_gain=float_slider(min_value=0, max_value=1, step=0.05, value=0.4, description="8ve"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Signal addition is the core rule behind interference and timbre. Once you are comfortable thinking in sums of simple waves, modulation and filtering become much less mysterious."""
        ),
    ],
    ROOT / "02-sampling-and-digital/01-analog-to-digital.ipynb": [
        md(
            """# Analog to Digital

This notebook introduces sampling: turning a continuous-time waveform into discrete values at fixed time intervals. It connects sample spacing to what a converter can and cannot represent."""
        ),
        code(COMMON_SETUP),
        md(
            """## Sampling a Sine Wave

Sampling does not store every point on the analog curve. It stores a sequence of values taken every `1/fs` seconds."""
        ),
        code(
            """analog_fs = 200_000
signal_freq = 1200
t_analog = np.arange(0, 0.008, 1 / analog_fs)
analog = np.cos(2 * np.pi * signal_freq * t_analog)

fig, ax = plt.subplots(figsize=(10, 3.5))

def update_sampling(fs_sample=10_000):
    ax.clear()
    t_samples = np.arange(0, t_analog[-1], 1 / fs_sample)
    samples = np.cos(2 * np.pi * signal_freq * t_samples)
    ax.plot(t_analog * 1e3, analog, label="Analog reference")
    ax.stem(t_samples * 1e3, samples, linefmt="tab:red", markerfmt="ro", basefmt=" ", label="Samples")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Sampling at {fs_sample:.0f} Hz")
    ax.legend()
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_sampling,
    fs_sample=float_slider(min_value=2500, max_value=30000, step=500, value=10000, description="fs"),
)
display(controls)
"""
        ),
        md(
            """## Staircase Reconstruction

Real DACs reconstruct from discrete samples. A simple zero-order hold makes a staircase; an analog low-pass filter smooths that staircase into something closer to the original."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 3.5))

def update_reconstruction(fs_sample=8000):
    ax.clear()
    t_samples = np.arange(0, t_analog[-1], 1 / fs_sample)
    samples = np.cos(2 * np.pi * signal_freq * t_samples)
    ax.plot(t_analog * 1e3, analog, label="Analog reference", alpha=0.5)
    ax.step(t_samples * 1e3, samples, where="post", label="Zero-order hold", color="tab:orange")
    ax.plot(t_samples * 1e3, samples, "o", color="tab:red")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Staircase Reconstruction")
    ax.legend()
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_reconstruction,
    fs_sample=float_slider(min_value=2500, max_value=30000, step=500, value=8000, description="fs"),
)
display(controls)
"""
        ),
        md(
            """## Key Takeaway

Digital signals are not magic analog copies. They are time-spaced measurements, and everything about fidelity depends on how often you take them and how you reconstruct them later."""
        ),
    ],
    ROOT / "02-sampling-and-digital/02-aliasing.ipynb": [
        md(
            """# Aliasing

This notebook makes Nyquist audible. When the sample rate drops below twice the highest frequency present, the digitized signal folds into a false lower frequency instead of representing the original one."""
        ),
        code(COMMON_SETUP),
        md(
            """## Frequency Folding

Sampling only constrains frequency modulo the sample rate. That is why multiple analog frequencies can pass through the same sample sequence."""
        ),
        code(
            """tone_freq = 6000
duration = 1.0
audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_aliasing(fs_sample=20_000):
    axes[0].clear()
    axes[1].clear()
    t_hr = np.arange(0, 0.005, 1 / 200_000)
    analog = np.cos(2 * np.pi * tone_freq * t_hr)
    t_samples = np.arange(0, duration, 1 / fs_sample)
    sampled = np.cos(2 * np.pi * tone_freq * t_samples)
    play = normalize(resample_signal(sampled, fs_sample, 44_100))
    alias_freq = abs(((tone_freq + fs_sample / 2) % fs_sample) - fs_sample / 2)

    axes[0].plot(t_hr * 1e3, analog, label="Analog")
    axes[0].stem(t_samples[:40] * 1e3, sampled[:40], linefmt="tab:red", markerfmt="ro", basefmt=" ")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Samples at {fs_sample:.0f} Hz")
    axes[1].text(0.05, 0.7, f"Original tone: {tone_freq:.0f} Hz", transform=axes[1].transAxes, fontsize=12)
    axes[1].text(0.05, 0.5, f"Observed alias: {alias_freq:.0f} Hz", transform=axes[1].transAxes, fontsize=12)
    axes[1].text(0.05, 0.3, f"Nyquist limit: {fs_sample/2:.0f} Hz", transform=axes[1].transAxes, fontsize=12)
    axes[1].set_axis_off()
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, play, rate=44_100)

controls = widgets.interactive(
    update_aliasing,
    fs_sample=float_slider(min_value=4000, max_value=24000, step=500, value=20000, description="fs"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Same Samples, Different Analog Stories

The aliasing trap is that the discrete samples alone do not tell you which high-frequency source created them. Anti-alias filtering is what removes those impossible alternatives before sampling."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 3.5))

def update_alias_visual(fs_sample=8000, analog_freq=6000):
    ax.clear()
    t = np.arange(0, 0.006, 1 / 200_000)
    t_samples = np.arange(0, t[-1], 1 / fs_sample)
    analog = np.cos(2 * np.pi * analog_freq * t)
    samples = np.cos(2 * np.pi * analog_freq * t_samples)
    alias_freq = abs(((analog_freq + fs_sample / 2) % fs_sample) - fs_sample / 2)
    alias_curve = np.cos(2 * np.pi * alias_freq * t)
    ax.plot(t * 1e3, analog, label="Original analog")
    ax.plot(t * 1e3, alias_curve, "--", label="Aliased interpretation")
    ax.plot(t_samples * 1e3, samples, "o", label="Samples")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Multiple analog curves through the same samples")
    ax.legend()
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_alias_visual,
    fs_sample=float_slider(min_value=4000, max_value=16000, step=500, value=8000, description="fs"),
    analog_freq=float_slider(min_value=1000, max_value=12000, step=250, value=6000, description="f analog"),
)
display(controls)
"""
        ),
        md(
            """## Key Takeaway

Aliasing is not just distortion. It is a wrong answer that looks internally consistent. That is why anti-alias filtering and Nyquist discipline matter before the ADC ever takes a sample."""
        ),
    ],
    ROOT / "07-receiver-chain/01-superheterodyne.ipynb": [
        md(
            """# Superheterodyne Receiver

This notebook turns the receiver-chain concepts into a concrete signal path: RF filtering, mixing to an intermediate frequency, IF selection, and demodulation."""
        ),
        code(COMMON_SETUP),
        md(
            """## Why Convert to an IF?

Mixing shifts a desired channel from its RF location down to a fixed intermediate frequency where narrow, stable filters are easier to build and reason about."""
        ),
        code(
            """fs = 192_000
t = np.arange(0, 0.03, 1 / fs)
rf_stations = [
    (40_000, 0.9, 800),
    (55_000, 0.6, 1400),
    (72_000, 0.5, 2200),
]
rf_band = np.zeros_like(t)
for carrier, gain, audio_freq in rf_stations:
    rf_band += gain * am_modulate(np.cos(2 * np.pi * audio_freq * t), carrier_freq=carrier, fs=fs, mod_index=0.7)

fig, ax = plt.subplots(figsize=(10, 3.5))
plot_spectrum(rf_band, fs=fs, ax=ax, title="Three RF Channels in the Front End")
ax.set_xlim(0, 90_000)
ax.set_ylim(-110, 5)
plt.tight_layout()
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
IF_FREQ = 12_000

def update_tuning(lo_freq=52_000):
    mixer_lo = np.cos(2 * np.pi * lo_freq * t)
    mixed = rf_band * mixer_lo
    sos = signal.butter(5, [IF_FREQ - 4000, IF_FREQ + 4000], btype="band", fs=fs, output="sos")
    if_signal = signal.sosfilt(sos, mixed)
    demod = am_demodulate(if_signal * np.cos(2 * np.pi * IF_FREQ * t), fs=fs, audio_cutoff=4000)

    for ax in axes:
        ax.clear()
    plot_spectrum(rf_band, fs=fs, ax=axes[0], title="RF Band")
    axes[0].axvline(lo_freq, color="tab:red", linestyle="--", label="LO")
    axes[0].legend()
    plot_spectrum(mixed, fs=fs, ax=axes[1], title="Mixer Output")
    plot_spectrum(if_signal, fs=fs, ax=axes[2], title="Filtered IF")
    axes[0].set_xlim(0, 90_000)
    axes[1].set_xlim(0, 90_000)
    axes[2].set_xlim(0, 30_000)
    for ax in axes:
        ax.set_ylim(-110, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(demod, fs, 44_100), rate=44_100)

controls = widgets.interactive(
    update_tuning,
    lo_freq=float_slider(min_value=44_000, max_value=84_000, step=500, value=52_000, description="LO Hz"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

The receiver does not directly "listen" at RF and recover audio in one step. It shifts energy through stages, and each stage exists to make selection and demodulation easier."""
        ),
    ],
    ROOT / "07-receiver-chain/02-build-a-receiver.ipynb": [
        md(
            """# Build a Receiver

This notebook presents the receiver as a set of stages you can inspect: antenna input, mixer output, IF filter, and audio output. Instead of a clickable diagram, it uses a stage selector that exposes the waveform and spectrum at each point."""
        ),
        code(COMMON_SETUP),
        md(
            """## Stage-by-Stage Inspection

When you choose a stage below, you are looking at the same signal after different receiver operations have been applied."""
        ),
        code(
            """fs = 192_000
t = np.arange(0, 0.04, 1 / fs)
message = 0.8 * np.sin(2 * np.pi * 1500 * t) + 0.3 * np.sin(2 * np.pi * 500 * t)
rf = fm_modulate(message, carrier_freq=52_000, fs=fs, freq_dev=2500)
adjacent = 0.5 * fm_modulate(0.7 * np.sin(2 * np.pi * 2300 * t), carrier_freq=63_000, fs=fs, freq_dev=1800)
front_end = rf + adjacent
lo = np.cos(2 * np.pi * 40_000 * t)
mixed = front_end * lo
if_sos = signal.butter(5, [10_000, 14_000], btype="band", fs=fs, output="sos")
if_signal = signal.sosfilt(if_sos, mixed)
audio = fm_demodulate(if_signal, fs=fs, audio_cutoff=4000)

stages = {
    "RF front end": front_end,
    "Mixer output": mixed,
    "IF filter": if_signal,
    "Recovered audio": audio,
}
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_stage(stage_name="RF front end"):
    sig = stages[stage_name]
    axes[0].clear()
    axes[1].clear()
    plot_waveform(sig[:6000], fs=fs, ax=axes[0], title=stage_name)
    plot_spectrum(sig, fs=fs, ax=axes[1], title=f"{stage_name} spectrum")
    axes[1].set_xlim(0, 80_000 if stage_name != "Recovered audio" else 10_000)
    axes[1].set_ylim(-110, 5)
    fig.canvas.draw_idle()
    if stage_name == "Recovered audio":
        refresh_audio_widget(audio_out, resample_signal(sig, fs, 44_100), rate=44_100)
    else:
        with audio_out:
            audio_out.clear_output(wait=True)
            display(Markdown("Select **Recovered audio** to hear the demodulated result."))

controls = widgets.interactive(
    update_stage,
    stage_name=dropdown(options=list(stages.keys()), value="RF front end", description="Stage"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Receiver design is easier to understand when you stop thinking of a radio as a black box. Each stage changes what information is obvious and what information is suppressed."""
        ),
    ],
    ROOT / "08-transmitter-fundamentals/01-oscillators-and-plls.ipynb": [
        md(
            """# Oscillators and PLLs

This notebook covers how transmitters create stable RF. A voltage-controlled oscillator can wander; a phase-locked loop keeps it tied to a clean reference."""
        ),
        code(COMMON_SETUP),
        md(
            """## Locking a VCO to a Reference

The simplest intuition for a PLL is feedback: compare the VCO to a reference, measure the error, and nudge the VCO until the error shrinks."""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def simulate_pll(loop_gain=0.08):
    ref = 10_000.0
    target = 8 * ref
    freqs = []
    errors = []
    vco = 72_000.0
    for _ in range(150):
        error = target - vco
        vco += loop_gain * error
        freqs.append(vco)
        errors.append(error)
    return np.array(freqs), np.array(errors)

def update_pll(loop_gain=0.08):
    freqs, errors = simulate_pll(loop_gain=loop_gain)
    axes[0].clear()
    axes[1].clear()
    axes[0].plot(freqs)
    axes[0].axhline(80_000, color="tab:red", linestyle="--", label="Target")
    axes[0].set_title("VCO Frequency vs Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend()
    axes[1].plot(errors, color="tab:orange")
    axes[1].set_title("Phase/Frequency Error")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Error (Hz)")
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_pll,
    loop_gain=float_slider(min_value=0.01, max_value=0.25, step=0.01, value=0.08, description="Loop gain"),
)
display(controls)
"""
        ),
        md(
            """## Harmonics and Filtering

Power amplifiers and nonlinear stages create harmonics. Transmitters need output filtering so most of the power stays in the intended channel."""
        ),
        code(
            """fs = 200_000
t = np.arange(0, 0.01, 1 / fs)
fundamental = np.cos(2 * np.pi * 20_000 * t)
dirty = normalize(fundamental + 0.4 * np.cos(2 * np.pi * 40_000 * t) + 0.2 * np.cos(2 * np.pi * 60_000 * t))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_harmonics(poles=4):
    axes[0].clear()
    axes[1].clear()
    sos = signal.butter(poles, 25_000, btype="low", fs=fs, output="sos")
    cleaned = signal.sosfilt(sos, dirty)
    plot_spectrum(dirty, fs=fs, ax=axes[0], title="Before output filter")
    plot_spectrum(cleaned, fs=fs, ax=axes[1], title="After output filter")
    for ax in axes:
        ax.set_xlim(0, 80_000)
        ax.set_ylim(-100, 5)
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_harmonics,
    poles=int_slider(min_value=2, max_value=10, step=2, value=4, description="Poles"),
)
display(controls)
"""
        ),
        md(
            """## Key Takeaway

Transmitters need both frequency generation and spectral cleanup. Oscillators create the carrier, PLLs keep it where it belongs, and output filters keep harmonics from escaping."""
        ),
    ],
    ROOT / "08-transmitter-fundamentals/02-ctcss-and-signaling.ipynb": [
        md(
            """# CTCSS and Signaling

This notebook focuses on the composite baseband in land-mobile FM: voice plus a low-frequency sub-audible tone used for selective squelch."""
        ),
        code(COMMON_SETUP),
        code(VOICE_SETUP),
        md(
            """## Voice Plus Tone

CTCSS tones live below the normal voice passband. They are present in the transmitter baseband, but ordinary speaker filtering tends to hide them from the listener."""
        ),
        code(
            """voice_bp = signal.sosfilt(signal.butter(5, [300, 3000], btype="band", fs=WORK_FS, output="sos"), voice_work)
voice_bp = normalize(voice_bp)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
audio_out = audio_output_widget()

def update_ctcss(tone_freq=141.3, tone_gain=0.25):
    tone = tone_gain * np.cos(2 * np.pi * tone_freq * t_work)
    composite = normalize(voice_bp + tone)
    speaker_hp = signal.sosfilt(signal.butter(5, 300, btype="highpass", fs=WORK_FS, output="sos"), composite)
    axes[0].clear()
    axes[1].clear()
    plot_waveform(composite[:12000], fs=WORK_FS, ax=axes[0], title="Composite Baseband")
    plot_spectrum(composite, fs=WORK_FS, ax=axes[1], title="Composite Spectrum")
    axes[1].set_xlim(0, 4000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    with audio_out:
        audio_out.clear_output(wait=True)
        display(Markdown("**Composite audio (tone included)**"))
        display(audio_player(resample_signal(composite, WORK_FS, PLAY_FS), rate=PLAY_FS))
        display(Markdown("**Speaker-path audio (tone mostly removed)**"))
        display(audio_player(resample_signal(speaker_hp, WORK_FS, PLAY_FS), rate=PLAY_FS))

controls = widgets.interactive(
    update_ctcss,
    tone_freq=float_slider(min_value=67.0, max_value=254.1, step=0.1, value=141.3, description="Tone Hz"),
    tone_gain=float_slider(min_value=0.0, max_value=0.5, step=0.02, value=0.25, description="Tone gain"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Selective signaling works because transmitters can carry information outside the normal voice band while receivers decide which parts of the composite signal to route to the speaker and which parts to route to control logic."""
        ),
    ],
    ROOT / "09-antennas-and-propagation/01-antenna-basics.ipynb": [
        md(
            """# Antenna Basics

This notebook introduces antenna patterns, gain, and polarization by visualizing simple directional models rather than jumping straight into hardware details."""
        ),
        code(COMMON_SETUP),
        md(
            """## Radiation Patterns

Antenna gain does not create power from nowhere. It redistributes where the power goes."""
        ),
        code(
            """theta = np.linspace(0, 2 * np.pi, 1000)
patterns = {
    "Isotropic": np.ones_like(theta),
    "Dipole": np.abs(np.sin(theta)),
    "Ground plane": 0.6 + 0.4 * np.abs(np.sin(theta)),
    "Yagi": np.clip(np.cos(theta / 2), 0, None) ** 6,
}

fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, projection="polar")

def update_pattern(pattern_name="Dipole"):
    ax.clear()
    ax.plot(theta, patterns[pattern_name])
    ax.set_title(pattern_name)
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_pattern,
    pattern_name=dropdown(options=list(patterns.keys()), value="Dipole", description="Pattern"),
)
display(controls)
"""
        ),
        md(
            """## SWR Intuition

SWR comes from reflections caused by mismatch. Reflection coefficient magnitude `|Gamma|` maps directly to SWR:

$$SWR = \\frac{1 + |\\Gamma|}{1 - |\\Gamma|}$$"""
        ),
        code(
            """out = widgets.Output()

def update_swr(gamma=0.2):
    with out:
        out.clear_output(wait=True)
        swr = (1 + gamma) / max(1 - gamma, 1e-9)
        print(f"Reflection coefficient magnitude: {gamma:.2f}")
        print(f"SWR: {swr:.2f}:1")

controls = widgets.interactive(
    update_swr,
    gamma=float_slider(min_value=0, max_value=0.95, step=0.01, value=0.2, description="|Gamma|"),
)
display(controls, out)
"""
        ),
        md(
            """## Key Takeaway

Antennas are spatial filters. Pattern and matching determine where power goes and how efficiently it moves between the transmitter, feedline, and free space."""
        ),
    ],
    ROOT / "09-antennas-and-propagation/02-propagation.ipynb": [
        md(
            """# Propagation

This notebook focuses on what happens after power leaves the antenna: free-space loss, Fresnel zone clearance, and how frequency changes the geometry of a path."""
        ),
        code(COMMON_SETUP),
        md(
            """## Free-Space Path Loss Across Bands

Higher frequency means shorter wavelength, which increases free-space loss for the same path length."""
        ),
        code(
            """distances = np.logspace(-1, 2, 300)
freqs_mhz = {
    "HF 7 MHz": 7,
    "VHF 146 MHz": 146,
    "UHF 462 MHz": 462,
    "ISM 915 MHz": 915,
    "Wi-Fi 2400 MHz": 2400,
}

fig, ax = plt.subplots(figsize=(10, 4))
for label, freq in freqs_mhz.items():
    ax.plot(distances, fspl_db(distances, freq), label=label)
ax.set_xscale("log")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("FSPL (dB)")
ax.set_title("Path loss by frequency")
ax.legend()
plt.tight_layout()
"""
        ),
        md(
            """## Fresnel Zone Width

Line of sight is not enough by itself. Objects intruding into the Fresnel zone can still damage the path."""
        ),
        code(
            """fig, ax = plt.subplots(figsize=(10, 3.5))

def fresnel_radius(d1_km, d2_km, freq_mhz):
    wavelength = 300 / freq_mhz
    d1 = d1_km * 1000
    d2 = d2_km * 1000
    return np.sqrt(wavelength * d1 * d2 / (d1 + d2))

def update_fresnel(total_distance_km=10.0, freq_mhz=462.0):
    ax.clear()
    x = np.linspace(0, total_distance_km, 400)
    radius = fresnel_radius(total_distance_km / 2, total_distance_km / 2, freq_mhz)
    profile = radius * np.sqrt(np.clip(1 - ((x - total_distance_km / 2) / (total_distance_km / 2)) ** 2, 0, None))
    ax.plot(x, profile, label="1st Fresnel zone")
    ax.plot(x, -profile, color="tab:blue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Path distance (km)")
    ax.set_ylabel("Radius (m)")
    ax.set_title(f"Mid-path Fresnel radius: {radius:.2f} m")
    ax.legend()
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_fresnel,
    total_distance_km=float_slider(min_value=1, max_value=50, step=1, value=10, description="Distance km"),
    freq_mhz=float_slider(min_value=30, max_value=2400, step=10, value=462, description="Freq MHz"),
)
display(controls)
"""
        ),
        md(
            """## Key Takeaway

Propagation is geometry plus wavelength. Distance, frequency, and clearance determine whether a path is clean, marginal, or doomed."""
        ),
    ],
    ROOT / "09-antennas-and-propagation/03-link-budgets.ipynb": [
        md(
            """# Link Budgets

This notebook turns propagation and gain into a practical calculator. It answers the engineering question: given a transmitter, antennas, path, and receiver sensitivity, is the link likely to work?"""
        ),
        code(COMMON_SETUP),
        md(
            """## Budget Arithmetic

In dB form, a basic link budget is:

`Pr = Pt + Gt + Gr - FSPL - losses`

Compare received power against sensitivity to get margin."""
        ),
        code(
            """presets = {
    "Baofeng to Baofeng": dict(tx_power_w=5, tx_gain_dbi=0, rx_gain_dbi=0, freq_mhz=462, distance_km=3, sensitivity_dbm=-118),
    "Baofeng to repeater": dict(tx_power_w=5, tx_gain_dbi=0, rx_gain_dbi=6, freq_mhz=462, distance_km=20, sensitivity_dbm=-120),
    "Meshtastic node to node": dict(tx_power_w=1, tx_gain_dbi=2, rx_gain_dbi=2, freq_mhz=915, distance_km=5, sensitivity_dbm=-130),
}

def watts_to_dbm(watts):
    return 10 * np.log10(watts * 1000)

out = widgets.Output()

def update_budget(preset="Baofeng to Baofeng", tx_power_w=5.0, tx_gain_dbi=0.0, rx_gain_dbi=0.0, freq_mhz=462.0, distance_km=3.0, sensitivity_dbm=-118.0):
    if preset in presets:
        vals = presets[preset]
        tx_power_w = vals["tx_power_w"]
        tx_gain_dbi = vals["tx_gain_dbi"]
        rx_gain_dbi = vals["rx_gain_dbi"]
        freq_mhz = vals["freq_mhz"]
        distance_km = vals["distance_km"]
        sensitivity_dbm = vals["sensitivity_dbm"]
    tx_dbm = watts_to_dbm(tx_power_w)
    path_loss = fspl_db(distance_km, freq_mhz)
    rx_dbm = tx_dbm + tx_gain_dbi + rx_gain_dbi - path_loss
    margin = rx_dbm - sensitivity_dbm
    verdict = "Likely workable" if margin > 10 else "Marginal" if margin > 0 else "Unlikely"
    with out:
        out.clear_output(wait=True)
        print(f"Tx power: {tx_dbm:.2f} dBm")
        print(f"Path loss: {path_loss:.2f} dB")
        print(f"Received power: {rx_dbm:.2f} dBm")
        print(f"Margin above sensitivity: {margin:.2f} dB")
        print(f"Verdict: {verdict}")

controls = widgets.interactive(
    update_budget,
    preset=dropdown(options=list(presets.keys()), value="Baofeng to Baofeng", description="Preset"),
    tx_power_w=float_slider(min_value=0.1, max_value=50, step=0.1, value=5, description="Tx W"),
    tx_gain_dbi=float_slider(min_value=-5, max_value=15, step=0.5, value=0, description="Tx dBi"),
    rx_gain_dbi=float_slider(min_value=-5, max_value=15, step=0.5, value=0, description="Rx dBi"),
    freq_mhz=float_slider(min_value=30, max_value=2400, step=1, value=462, description="MHz"),
    distance_km=float_slider(min_value=0.1, max_value=100, step=0.1, value=3, description="km"),
    sensitivity_dbm=float_slider(min_value=-140, max_value=-80, step=1, value=-118, description="Sens dBm"),
)
display(controls, out)
"""
        ),
        md(
            """## Key Takeaway

Link budgets keep RF decisions honest. You can disagree about terrain assumptions or fading margin, but the arithmetic forces every assumption onto the table."""
        ),
    ],
    ROOT / "11-digital-modulation/01-ask-fsk-psk-qam.ipynb": [
        md(
            """# ASK, FSK, PSK, and QAM

This notebook introduces common digital modulation families by mapping a short bit pattern into amplitude, frequency, phase, or combined amplitude/phase changes."""
        ),
        code(COMMON_SETUP),
        md(
            """## One Bitstream, Many Encodings

Digital modulation is just a decision about what physical parameter will represent symbols."""
        ),
        code(
            """bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
symbol_rate = 200
fs = 20_000
samples_per_symbol = fs // symbol_rate
t_symbol = np.arange(samples_per_symbol) / fs

def build_ask():
    return np.concatenate([(0.3 + 0.7 * bit) * np.cos(2 * np.pi * 1200 * t_symbol) for bit in bits])

def build_fsk():
    return np.concatenate([np.cos(2 * np.pi * (900 if bit == 0 else 1500) * t_symbol) for bit in bits])

def build_bpsk():
    return np.concatenate([np.cos(2 * np.pi * 1200 * t_symbol + (0 if bit == 1 else np.pi)) for bit in bits])

def build_qpsk():
    pairs = bits.reshape(-1, 2)
    phases = {(0, 0): 5 * np.pi / 4, (0, 1): 3 * np.pi / 4, (1, 1): np.pi / 4, (1, 0): 7 * np.pi / 4}
    return np.concatenate([np.cos(2 * np.pi * 1200 * t_symbol + phases[tuple(pair)]) for pair in pairs])

signals = {
    "ASK": build_ask(),
    "FSK": build_fsk(),
    "BPSK": build_bpsk(),
    "QPSK": build_qpsk(),
}
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_scheme(scheme="ASK"):
    sig = signals[scheme]
    axes[0].clear()
    axes[1].clear()
    plot_waveform(sig[:2000], fs=fs, ax=axes[0], title=f"{scheme} waveform")
    plot_spectrum(sig, fs=fs, ax=axes[1], title=f"{scheme} spectrum")
    axes[1].set_xlim(0, 4000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, normalize(sig), rate=fs)

controls = widgets.interactive(
    update_scheme,
    scheme=dropdown(options=list(signals.keys()), value="ASK", description="Scheme"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Digital modulation is still modulation. The difference is that the transmitter chooses from a finite symbol alphabet instead of letting the parameter vary continuously."""
        ),
    ],
    ROOT / "11-digital-modulation/02-constellation-diagrams.ipynb": [
        md(
            """# Constellation Diagrams

This notebook turns digital modulation into points on the I/Q plane. As noise rises, clusters spread, decision boundaries blur, and bit errors become more likely."""
        ),
        code(COMMON_SETUP),
        md(
            """## QPSK and 16-QAM Under Noise

Constellation diagrams are just scatter plots of complex symbols. Their geometry makes it easy to see why denser alphabets demand more SNR."""
        ),
        code(
            """rng = np.random.default_rng(42)
qpsk = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
qam16 = np.array([x + 1j * y for x in (-3, -1, 1, 3) for y in (-3, -1, 1, 3)]) / np.sqrt(10)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

def update_constellation(noise_sigma=0.1):
    for ax in axes:
        ax.clear()
    qpsk_samples = rng.choice(qpsk, size=600) + noise_sigma * (rng.normal(size=600) + 1j * rng.normal(size=600))
    qam_samples = rng.choice(qam16, size=600) + noise_sigma * (rng.normal(size=600) + 1j * rng.normal(size=600))
    axes[0].scatter(qpsk_samples.real, qpsk_samples.imag, s=12, alpha=0.6)
    axes[1].scatter(qam_samples.real, qam_samples.imag, s=12, alpha=0.6, color="tab:orange")
    axes[0].set_title("QPSK")
    axes[1].set_title("16-QAM")
    for ax in axes:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
    axes[1].set_xlim(-1.8, 1.8)
    axes[1].set_ylim(-1.8, 1.8)
    fig.canvas.draw_idle()

controls = widgets.interactive(
    update_constellation,
    noise_sigma=float_slider(min_value=0.01, max_value=0.6, step=0.01, value=0.1, description="Noise"),
)
display(controls)
"""
        ),
        md(
            """## BER Intuition

When noise spreads symbols across the wrong decision region, a bit error occurs. Dense constellations pack more bits per symbol, but their boundaries are closer together."""
        ),
        code(
            """out = widgets.Output()

def estimate_qpsk_ber(noise_sigma=0.1, n=20_000):
    bits = rng.integers(0, 2, size=(n, 2))
    mapping = {(0, 0): -1 - 1j, (0, 1): -1 + 1j, (1, 1): 1 + 1j, (1, 0): 1 - 1j}
    symbols = np.array([mapping[tuple(pair)] for pair in bits]) / np.sqrt(2)
    noisy = symbols + noise_sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    decided = np.column_stack((noisy.real > 0, noisy.imag > 0)).astype(int)
    remap = {(0, 0): np.array([0, 0]), (0, 1): np.array([0, 1]), (1, 1): np.array([1, 1]), (1, 0): np.array([1, 0])}
    decoded = np.array([remap[tuple(pair)] for pair in decided])
    ber = np.mean(bits != decoded)
    with out:
        out.clear_output(wait=True)
        print(f"Estimated QPSK BER: {ber:.5f}")

controls = widgets.interactive(
    estimate_qpsk_ber,
    noise_sigma=float_slider(min_value=0.01, max_value=0.8, step=0.01, value=0.1, description="Noise"),
)
display(controls, out)
"""
        ),
        md(
            """## Key Takeaway

Constellations make digital tradeoffs visible. More bits per symbol improve efficiency, but they also make every noise burst and phase error more expensive."""
        ),
    ],
    ROOT / "11-digital-modulation/03-lora-chirp-spread-spectrum.ipynb": [
        md(
            """# LoRa Chirp Spread Spectrum

This notebook introduces LoRa-style chirp spread spectrum using synthetic chirps. It focuses on the visual intuition: spreading factor changes chirp duration, symbol rate, and the time-bandwidth tradeoff."""
        ),
        code(COMMON_SETUP),
        md(
            """## Chirps as Symbols

Instead of holding a fixed carrier state for each symbol, LoRa sweeps frequency across the channel. That makes the signal look like a diagonal trace in a spectrogram."""
        ),
        code(
            """def lora_like_chirp(sf=7, bw=125_000, fs=500_000):
    symbol_time = (2 ** sf) / bw
    t = np.arange(0, symbol_time, 1 / fs)
    chirp = signal.chirp(t, f0=-bw / 2, f1=bw / 2, t1=symbol_time, method="linear")
    return t, chirp

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
audio_out = audio_output_widget()

def update_lora(sf=7, bw=125_000):
    t, chirp = lora_like_chirp(sf=int(sf), bw=bw)
    axes[0].clear()
    axes[1].clear()
    plot_waveform(chirp[:3000], fs=500_000, ax=axes[0], title=f"Chirp waveform, SF{sf}")
    plot_spectrogram(chirp, fs=500_000, ax=axes[1], title="Chirp spectrogram", nperseg=512, noverlap=384)
    axes[1].set_ylim(0, bw / 2)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(chirp, 500_000, 44_100), rate=44_100)

controls = widgets.interactive(
    update_lora,
    sf=int_slider(min_value=7, max_value=12, step=1, value=7, description="SF"),
    bw=float_slider(min_value=62_500, max_value=500_000, step=62_500, value=125_000, description="BW", readout_format=".0f"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Symbol Time vs Spreading Factor

Increasing spreading factor makes the chirp longer, which improves sensitivity but lowers data rate."""
        ),
        code(
            """out = widgets.Output()

def update_tradeoff(sf=7, bw=125_000):
    symbol_time = (2 ** sf) / bw
    raw_symbol_rate = 1 / symbol_time
    with out:
        out.clear_output(wait=True)
        print(f"Spreading factor: SF{sf}")
        print(f"Bandwidth: {bw/1000:.1f} kHz")
        print(f"Symbol time: {symbol_time * 1e3:.3f} ms")
        print(f"Symbol rate: {raw_symbol_rate:.2f} symbols/s")

controls = widgets.interactive(
    update_tradeoff,
    sf=int_slider(min_value=7, max_value=12, step=1, value=7, description="SF"),
    bw=float_slider(min_value=62_500, max_value=500_000, step=62_500, value=125_000, description="BW", readout_format=".0f"),
)
display(controls, out)
"""
        ),
        md(
            """## Key Takeaway

LoRa's chirps trade speed for resilience. The longer the chirp relative to the bandwidth, the easier it is to dig weak energy out of noise, but the fewer symbols you send per second."""
        ),
    ],
}


for path, cells in NOTEBOOKS.items():
    write_notebook(path, cells)
    print(f"Wrote {path.relative_to(ROOT)}")
