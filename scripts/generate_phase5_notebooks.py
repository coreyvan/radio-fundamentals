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


PRACTICAL_SETUP = """CAPTURE_ROOT = ROOT / "assets" / "local"
capture_status = probe_rtlsdr()
display(Markdown(
    f"**RTL-SDR status:** installed={capture_status['installed']}, "
    f"available={capture_status['available']}. {capture_status['message']}"
))
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
    ROOT / "12-practical-projects/01-fm-receiver-from-iq.ipynb": [
        md(
            """# FM Receiver from IQ Samples

This notebook builds a simple software receiver around either a local IQ capture or a synthetic fallback signal. It is the first practical notebook that runs end-to-end from RF-ish samples to recovered audio."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        md(
            """## Capture Source

If `assets/local/fm_receiver_iq.npz` exists, it will be used. Otherwise this notebook synthesizes a narrowband FM signal at an offset inside a complex baseband recording."""
        ),
        code(
            """capture_path = CAPTURE_ROOT / "fm_receiver_iq.npz"
if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 240_000
    t = np.arange(0, 3.0, 1 / fs_iq)
    message = normalize(
        0.7 * np.sin(2 * np.pi * 700 * t)
        + 0.3 * np.sin(2 * np.pi * 1400 * t)
        + 0.2 * np.sin(2 * np.pi * 2200 * t)
    )
    iq = synthesize_fm_iq(message, fs=fs_iq, carrier_offset=45_000, freq_dev=2500)
    iq += 0.15 * synthesize_fm_iq(0.5 * np.sin(2 * np.pi * 1200 * t), fs=fs_iq, carrier_offset=-30_000, freq_dev=1800)
    iq = normalize(iq)
    print("Using synthetic FM IQ fallback.")
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(2, 2, figsize=(13, 7))

def update_receiver(tune_offset=45_000.0, channel_bw=12_500.0):
    shifted = complex_mix_down(iq, fs=fs_iq, freq_shift=tune_offset)
    filtered = lowpass_filter(shifted, cutoff_hz=channel_bw / 2, fs=fs_iq, order=5)
    audio = fm_demodulate_iq(filtered, fs=fs_iq, audio_cutoff=4000)

    for ax in axes.flat:
        ax.clear()
    plot_spectrum(iq.real, fs=fs_iq, ax=axes[0, 0], title="I-channel spectrum")
    plot_spectrum(np.real(shifted), fs=fs_iq, ax=axes[0, 1], title="After tuning")
    plot_waveform(np.real(filtered[:12000]), fs=fs_iq, ax=axes[1, 0], title="Filtered baseband")
    plot_spectrum(audio, fs=fs_iq, ax=axes[1, 1], title="Recovered audio")
    axes[0, 0].set_xlim(0, 120_000)
    axes[0, 1].set_xlim(0, 80_000)
    axes[1, 1].set_xlim(0, 6000)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_ylim(-100, 5)
    fig.canvas.draw_idle()
    refresh_audio_widget(audio_out, resample_signal(audio, fs_iq, 44_100), rate=44_100)

controls = widgets.interactive(
    update_receiver,
    tune_offset=float_slider(min_value=-80_000, max_value=80_000, step=1000, value=45_000, description="Tune Hz", readout_format=".0f"),
    channel_bw=float_slider(min_value=8_000, max_value=30_000, step=500, value=12_500, description="BW Hz", readout_format=".0f"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

An SDR receiver is just DSP applied to captured samples. Once you can tune, filter, and demodulate a file, live hardware becomes an input source problem rather than a fundamentally new architecture."""
        ),
    ],
    ROOT / "12-practical-projects/02-decode-aprs.ipynb": [
        md(
            """# Decode APRS

This notebook walks through the APRS receive chain using local IQ samples if available, or a synthetic Bell 202 AFSK-over-FM fallback if not. The goal is to make the FM -> AFSK -> symbol decision path visible."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        md(
            """## Capture Source and Synthetic Frame

Local path: `assets/local/aprs_afsk_iq.npz`. Synthetic fallback encodes a simple alternating Bell 202 tone sequence that stands in for an AX.25 frame."""
        ),
        code(
            """capture_path = CAPTURE_ROOT / "aprs_afsk_iq.npz"
symbol_rate = 1200
mark = 1200
space = 2200

if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 96_000
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1] * 30)
    samples_per_symbol = fs_iq // symbol_rate
    t_sym = np.arange(samples_per_symbol) / fs_iq
    afsk = np.concatenate([
        np.cos(2 * np.pi * (mark if bit else space) * t_sym) for bit in bits
    ])
    iq = synthesize_fm_iq(normalize(afsk), fs=fs_iq, carrier_offset=18_000, freq_dev=3000)
    print("Using synthetic APRS-like Bell 202 fallback.")
"""
        ),
        code(
            """audio = fm_demodulate_iq(complex_mix_down(iq, fs_iq, 18_000), fs=fs_iq, audio_cutoff=3500)
f_mark, p_mark = power_spectrum(audio, fs_iq, nfft=8192)

samples_per_symbol = fs_iq // symbol_rate
usable = len(audio) // samples_per_symbol
audio_symbols = audio[: usable * samples_per_symbol].reshape(usable, samples_per_symbol)
t_sym = np.arange(samples_per_symbol) / fs_iq
mark_ref = np.cos(2 * np.pi * mark * t_sym)
space_ref = np.cos(2 * np.pi * space * t_sym)
mark_energy = np.abs(audio_symbols @ mark_ref)
space_energy = np.abs(audio_symbols @ space_ref)
decoded_bits = (mark_energy > space_energy).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
plot_waveform(audio[:12000], fs=fs_iq, ax=axes[0], title="Recovered AFSK audio")
plot_spectrum(audio, fs=fs_iq, ax=axes[1], title="Bell 202 tones")
axes[1].set_xlim(0, 3000)
axes[1].set_ylim(-100, 5)
axes[2].plot(decoded_bits[:80], drawstyle="steps-post")
axes[2].set_title("Detected bits")
axes[2].set_xlabel("Symbol index")
axes[2].set_ylim(-0.2, 1.2)
plt.tight_layout()

display(Markdown(f"**First 64 decoded bits:** `{''.join(map(str, decoded_bits[:64]))}`"))
display(audio_player(resample_signal(audio, fs_iq, 44_100), rate=44_100))
"""
        ),
        md(
            """## Key Takeaway

APRS decoding is a chain of simpler problems. First recover audio, then separate mark and space energy, then turn symbol decisions into bits and frames."""
        ),
    ],
    ROOT / "12-practical-projects/03-analyze-baofeng-tx.ipynb": [
        md(
            """# Analyze a Baofeng Transmission

This notebook measures deviation, CTCSS energy, and spectral occupancy from a local transmission capture if available. Without a capture, it uses a synthetic stand-in that behaves like a narrowband FM handheld transmission."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        code(
            """capture_path = CAPTURE_ROOT / "baofeng_tx_iq.npz"
if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 240_000
    t = np.arange(0, 3.0, 1 / fs_iq)
    voice = normalize(0.7 * np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 200 * t))
    composite = normalize(voice + 0.15 * np.cos(2 * np.pi * 141.3 * t))
    iq = synthesize_fm_iq(composite, fs=fs_iq, carrier_offset=35_000, freq_dev=2500)
    print("Using synthetic handheld-transmission fallback.")
"""
        ),
        code(
            """baseband = complex_mix_down(iq, fs_iq, 35_000)
audio = fm_demodulate_iq(baseband, fs=fs_iq, audio_cutoff=4000)
inst_freq = np.angle(baseband[1:] * np.conj(baseband[:-1])) * fs_iq / (2 * np.pi)
peak_dev = np.max(np.abs(inst_freq - np.mean(inst_freq)))

freqs, spectrum = power_spectrum(audio, fs_iq, nfft=8192)
ctcss_mask = (freqs >= 60) & (freqs <= 260)
ctcss_freq = freqs[ctcss_mask][np.argmax(spectrum[ctcss_mask])]
occupied_bw = freqs[np.where(spectrum > (np.max(spectrum) - 26))[0][-1]]

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
plot_spectrum(np.real(baseband), fs=fs_iq, ax=axes[0], title="Baseband spectrum")
axes[0].set_xlim(0, 50_000)
axes[0].set_ylim(-100, 5)
plot_spectrum(audio, fs=fs_iq, ax=axes[1], title="Recovered audio spectrum")
axes[1].set_xlim(0, 4000)
axes[1].set_ylim(-100, 5)
axes[2].hist(inst_freq - np.mean(inst_freq), bins=100, color="tab:orange")
axes[2].set_title("Instantaneous frequency spread")
axes[2].set_xlabel("Deviation (Hz)")
plt.tight_layout()

display(Markdown(f"**Estimated peak deviation:** {peak_dev:.0f} Hz"))
display(Markdown(f"**Strongest sub-audible component:** {ctcss_freq:.1f} Hz"))
display(Markdown(f"**Approximate occupied audio-side bandwidth at -26 dB:** {occupied_bw:.0f} Hz"))
"""
        ),
        md(
            """## Key Takeaway

Compliance questions turn into measurements once you have a capture: estimate deviation from instantaneous frequency, find tone energy in the low audio band, and inspect occupied bandwidth against your assumptions."""
        ),
    ],
    ROOT / "12-practical-projects/04-meshtastic-signal-analysis.ipynb": [
        md(
            """# Meshtastic Signal Analysis

This notebook looks at LoRa-like chirp captures. It accepts a local Meshtastic capture if present, but otherwise synthesizes a packet-like burst so the spectrogram and timing measurements remain available."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        code(
            """capture_path = CAPTURE_ROOT / "meshtastic_iq.npz"
if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 500_000
    chirps = []
    for sf in [7, 7, 9, 10]:
        symbol_time = (2 ** sf) / 125_000
        t = np.arange(0, symbol_time, 1 / fs_iq)
        chirp = signal.chirp(t, f0=-62_500, f1=62_500, t1=t[-1], method="linear")
        chirps.append(np.exp(1j * np.angle(signal.hilbert(chirp))))
        chirps.append(np.zeros(int(0.02 * fs_iq), dtype=np.complex128))
    iq = np.concatenate(chirps)
    print("Using synthetic LoRa-like chirp burst fallback.")
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_waveform(np.real(iq[:20000]), fs=fs_iq, ax=axes[0], title="IQ real part")
plot_spectrogram(np.real(iq), fs=fs_iq, ax=axes[1], title="Chirp spectrogram", nperseg=1024, noverlap=768)
axes[1].set_ylim(0, 150_000)
plt.tight_layout()

ener = np.abs(iq)
active = ener > (0.2 * np.max(ener))
burst_duration_ms = np.sum(active) / fs_iq * 1e3
display(Markdown(f"**Approximate active burst time:** {burst_duration_ms:.2f} ms"))
"""
        ),
        md(
            """## Key Takeaway

Meshtastic-style signals are easiest to reason about in the time-frequency plane. The chirp structure is the feature, not an implementation detail."""
        ),
    ],
    ROOT / "12-practical-projects/05-decode-pocsag.ipynb": [
        md(
            """# Decode POCSAG

This notebook uses a local pager capture if you have one, otherwise it synthesizes a simple 2-FSK burst and walks through symbol recovery. The decoding is intentionally lightweight and focused on signal intuition."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        code(
            """capture_path = CAPTURE_ROOT / "pocsag_iq.npz"
symbol_rate = 1200

if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 96_000
    bits = np.array(([1, 0, 1, 0, 0, 1, 1, 0] * 40), dtype=int)
    samples_per_symbol = fs_iq // symbol_rate
    freq_steps = np.repeat(np.where(bits == 1, 2400, -2400), samples_per_symbol)
    phase = 2 * np.pi * np.cumsum(freq_steps) / fs_iq
    iq = np.exp(1j * phase)
    print("Using synthetic 2-FSK pager fallback.")
"""
        ),
        code(
            """discriminator = np.angle(iq[1:] * np.conj(iq[:-1])) * fs_iq / (2 * np.pi)
discriminator = np.concatenate([discriminator, discriminator[-1:]])
samples_per_symbol = fs_iq // symbol_rate
usable = len(discriminator) // samples_per_symbol
symbol_values = discriminator[: usable * samples_per_symbol].reshape(usable, samples_per_symbol).mean(axis=1)
decoded = (symbol_values > 0).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
plot_waveform(discriminator[:12000], fs=fs_iq, ax=axes[0], title="FSK discriminator output")
plot_spectrum(discriminator, fs=fs_iq, ax=axes[1], title="Baseband spectrum")
axes[1].set_xlim(0, 5000)
axes[1].set_ylim(-100, 5)
axes[2].plot(decoded[:100], drawstyle="steps-post")
axes[2].set_ylim(-0.2, 1.2)
axes[2].set_title("Decoded bits")
plt.tight_layout()

display(Markdown(f"**First 80 bits:** `{''.join(map(str, decoded[:80]))}`"))
"""
        ),
        md(
            """## Key Takeaway

POCSAG-style FSK decoding starts the same way as many other digital radio tasks: convert frequency changes into a real-valued discriminator output, average over symbol periods, and then make decisions."""
        ),
    ],
    ROOT / "13-test-and-measurement/01-rtlsdr-spectrum-analyzer.ipynb": [
        md(
            """# RTL-SDR as a Spectrum Analyzer

This notebook turns either a local IQ sweep capture or a synthetic wideband signal into a spectrum-viewing workflow. Live SDR use is optional and explicitly gated by `probe_rtlsdr()`."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        code(
            """capture_path = CAPTURE_ROOT / "spectrum_sweep_iq.npz"
if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 240_000
    t = np.arange(0, 1.0, 1 / fs_iq)
    iq = (
        1.0 * np.exp(1j * 2 * np.pi * 20_000 * t)
        + 0.5 * np.exp(1j * 2 * np.pi * -40_000 * t)
        + 0.25 * np.exp(1j * 2 * np.pi * 70_000 * t)
    )
    iq += 0.03 * (np.random.default_rng(42).normal(size=len(t)) + 1j * np.random.default_rng(43).normal(size=len(t)))
    print("Using synthetic wideband spectrum fallback.")
"""
        ),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 4))
freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1 / fs_iq))
spectrum = np.fft.fftshift(np.fft.fft(iq))
axes[0].plot(freqs, 20 * np.log10(np.abs(spectrum) / len(iq) + 1e-12))
axes[0].set_title("Wideband spectrum")
axes[0].set_xlabel("Frequency offset (Hz)")
axes[0].set_ylabel("Magnitude (dB)")
axes[0].set_ylim(-120, 5)
plot_spectrogram(np.real(iq), fs=fs_iq, ax=axes[1], title="Waterfall-like view", nperseg=2048, noverlap=1536)
axes[1].set_ylim(0, fs_iq / 2)
plt.tight_layout()
"""
        ),
        md(
            """## Key Takeaway

An RTL-SDR plus FFTs gives you the core behavior of a spectrum analyzer. The limitation is dynamic range and calibration, not the underlying idea."""
        ),
    ],
    ROOT / "13-test-and-measurement/02-measuring-deviation.ipynb": [
        md(
            """# Measuring Deviation

This notebook measures FM deviation from either a local tone test capture or a synthetic fallback. The main metric is the instantaneous frequency swing around the carrier."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        code(
            """capture_path = CAPTURE_ROOT / "deviation_test_iq.npz"
tone_freq = 1000

if capture_path.exists():
    fs_iq, iq = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    fs_iq = 240_000
    t = np.arange(0, 2.0, 1 / fs_iq)
    tone = np.cos(2 * np.pi * tone_freq * t)
    iq = synthesize_fm_iq(tone, fs=fs_iq, carrier_offset=0, freq_dev=2500)
    print("Using synthetic FM tone fallback.")
"""
        ),
        code(
            """inst_freq = np.angle(iq[1:] * np.conj(iq[:-1])) * fs_iq / (2 * np.pi)
inst_freq = np.concatenate([inst_freq, inst_freq[-1:]])
inst_freq = inst_freq - np.mean(inst_freq)
peak_dev = np.max(np.abs(inst_freq))
rms_dev = np.sqrt(np.mean(inst_freq**2))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
plot_waveform(inst_freq[:15000], fs=fs_iq, ax=axes[0], title="Instantaneous frequency")
plot_spectrum(inst_freq, fs=fs_iq, ax=axes[1], title="Deviation spectrum")
axes[1].set_xlim(0, 5000)
axes[1].set_ylim(-100, 5)
plt.tight_layout()

display(Markdown(f"**Peak deviation:** {peak_dev:.1f} Hz"))
display(Markdown(f"**RMS deviation:** {rms_dev:.1f} Hz"))
"""
        ),
        md(
            """## Key Takeaway

Deviation is directly measurable from phase progression in IQ data. You do not need a special black-box meter if you can compute instantaneous frequency robustly."""
        ),
    ],
    ROOT / "13-test-and-measurement/03-measuring-sensitivity.ipynb": [
        md(
            """# Measuring Sensitivity

This notebook uses a local weak-signal capture or a synthetic fallback to explore the threshold at which recovered audio becomes unreliable. It frames sensitivity as a measurement process rather than a single spec-sheet number."""
        ),
        code(COMMON_SETUP),
        code(PRACTICAL_SETUP),
        code(
            """capture_path = CAPTURE_ROOT / "sensitivity_test_iq.npz"
fs_iq = 96_000
t = np.arange(0, 2.0, 1 / fs_iq)
message = np.sin(2 * np.pi * 1000 * t)

if capture_path.exists():
    fs_iq, iq_clean = load_complex_capture(capture_path)
    print(f"Loaded local capture: {capture_path.name}, fs={fs_iq}")
else:
    iq_clean = synthesize_fm_iq(message, fs=fs_iq, carrier_offset=0, freq_dev=2000)
    print("Using synthetic weak-signal fallback.")
"""
        ),
        code(
            """audio_out = audio_output_widget()
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))

def update_sensitivity(snr_db=20.0):
    noisy_i = add_awgn(iq_clean.real, snr_db=snr_db, seed=42)[0]
    noisy_q = add_awgn(iq_clean.imag, snr_db=snr_db, seed=43)[0]
    noisy_iq = noisy_i + 1j * noisy_q
    audio = fm_demodulate_iq(noisy_iq, fs=fs_iq, audio_cutoff=4000)
    tone_power = np.mean(message**2)
    err_power = np.mean((message[: len(audio)] - audio[: len(message)]) ** 2)
    snr_est = 10 * np.log10((tone_power + 1e-12) / (err_power + 1e-12))

    for ax in axes:
        ax.clear()
    plot_waveform(audio[:12000], fs=fs_iq, ax=axes[0], title="Recovered audio")
    plot_spectrum(audio, fs=fs_iq, ax=axes[1], title="Recovered audio spectrum")
    axes[1].set_xlim(0, 5000)
    axes[1].set_ylim(-100, 5)
    fig.canvas.draw_idle()
    with audio_out:
        audio_out.clear_output(wait=True)
        display(Markdown(f"**Injected RF SNR:** {snr_db:.1f} dB"))
        display(Markdown(f"**Estimated audio-domain SNR proxy:** {snr_est:.2f} dB"))
        display(audio_player(resample_signal(audio, fs_iq, 44_100), rate=44_100))

controls = widgets.interactive(
    update_sensitivity,
    snr_db=float_slider(min_value=-5, max_value=30, step=1, value=20, description="RF SNR"),
)
display(controls, audio_out)
"""
        ),
        md(
            """## Key Takeaway

Sensitivity is a threshold problem. Whether you state it in dBm, SINAD, or recovered-audio quality, the workflow is the same: inject a weaker signal, measure the outcome, and define the fail point explicitly."""
        ),
    ],
}


for path, cells in NOTEBOOKS.items():
    write_notebook(path, cells)
    print(f"Wrote {path.relative_to(ROOT)}")
