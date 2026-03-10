import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

matplotlib.use("Agg")

import rf_utils


class RfUtilsTest(unittest.TestCase):
    @staticmethod
    def _aligned_correlation(a, b):
        a = np.asarray(a, dtype=np.float64) - np.mean(a)
        b = np.asarray(b, dtype=np.float64) - np.mean(b)
        corr = scipy_signal.correlate(b, a, mode="full")
        lags = scipy_signal.correlation_lags(len(b), len(a), mode="full")
        lag = lags[np.argmax(np.abs(corr))]
        if lag > 0:
            a = a[:-lag]
            b = b[lag:]
        elif lag < 0:
            a = a[-lag:]
            b = b[:lag]
        return abs(np.corrcoef(a, b)[0, 1])

    def test_generate_tone_and_spectrum_peak(self):
        fs = 48_000
        _, tone = rf_utils.generate_tone(freq=1_000, duration=0.05, fs=fs)
        freqs, mags_db = rf_utils.power_spectrum(tone, fs, nfft=8192)
        peak_freq = freqs[np.argmax(mags_db)]
        self.assertAlmostEqual(peak_freq, 1_000, delta=15)

    def test_am_round_trip_preserves_message_shape(self):
        fs = 48_000
        duration = 0.1
        t = np.arange(0, duration, 1 / fs)
        message = 0.6 * np.sin(2 * np.pi * 700 * t)
        modulated = rf_utils.am_modulate(message, carrier_freq=8_000, fs=fs, mod_index=0.7)
        demodulated = rf_utils.am_demodulate(modulated, fs=fs, audio_cutoff=2_000)
        corr = self._aligned_correlation(message, demodulated)
        self.assertGreater(corr, 0.95)

    def test_fm_round_trip_preserves_message_shape(self):
        fs = 96_000
        duration = 0.08
        t = np.arange(0, duration, 1 / fs)
        message = 0.8 * np.sin(2 * np.pi * 1_000 * t)
        modulated = rf_utils.fm_modulate(message, carrier_freq=20_000, fs=fs, freq_dev=2_500)
        demodulated = rf_utils.fm_demodulate(modulated, fs=fs, audio_cutoff=3_000)
        corr = self._aligned_correlation(message, demodulated)
        self.assertGreater(corr, 0.9)

    def test_add_awgn_returns_requested_shape_and_reasonable_noise(self):
        sig = np.ones(8_000)
        noisy, noise = rf_utils.add_awgn(sig, snr_db=20, seed=7)
        self.assertEqual(sig.shape, noisy.shape)
        self.assertEqual(sig.shape, noise.shape)
        measured_noise_power = np.mean(noise**2)
        self.assertAlmostEqual(measured_noise_power, 0.01, delta=0.004)

    def test_add_impulse_noise_injects_spikes(self):
        fs = 10_000
        sig = np.zeros(fs)
        noisy, noise = rf_utils.add_impulse_noise(sig, fs=fs, rate=40, amplitude=2.0, seed=3)
        self.assertEqual(sig.shape, noisy.shape)
        self.assertGreater(np.count_nonzero(noise), 0)
        self.assertGreater(np.max(np.abs(noise)), 0.5)

    def test_pre_and_de_emphasis_are_rough_inverse(self):
        fs = 48_000
        t = np.arange(0, 0.1, 1 / fs)
        sig = np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 2_000 * t)
        restored = rf_utils.de_emphasis(rf_utils.pre_emphasis(sig, fs=fs), fs=fs)
        corr = np.corrcoef(sig[100:], restored[100:])[0, 1]
        self.assertGreater(corr, 0.98)

    def test_load_audio_reads_wav_and_optionally_normalizes(self):
        fs = 22_050
        samples = np.array([0, 1000, -1000, 500], dtype=np.int16)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            wavfile.write(wav_path, fs, samples)
            loaded_fs, loaded = rf_utils.load_audio(wav_path, normalize_audio=True)
        self.assertEqual(loaded_fs, fs)
        self.assertAlmostEqual(np.max(np.abs(loaded)), 1.0, delta=1e-6)

    def test_fspl_db_matches_reference_value(self):
        loss = rf_utils.fspl_db(d_km=1.0, f_mhz=462.0)
        self.assertAlmostEqual(loss, 85.73, delta=0.05)

    def test_complex_mix_down_shifts_tone_to_baseband(self):
        fs = 48_000
        t = np.arange(0, 0.05, 1 / fs)
        iq = np.exp(1j * 2 * np.pi * 6_000 * t)
        shifted = rf_utils.complex_mix_down(iq, fs=fs, freq_shift=6_000)
        self.assertAlmostEqual(np.mean(np.abs(shifted - 1.0)), 0.0, delta=1e-2)

    def test_fm_demodulate_iq_recovers_message_shape(self):
        fs = 96_000
        t = np.arange(0, 0.08, 1 / fs)
        message = 0.7 * np.sin(2 * np.pi * 900 * t)
        iq = rf_utils.synthesize_fm_iq(message, fs=fs, carrier_offset=12_000, freq_dev=2_500)
        demod = rf_utils.fm_demodulate_iq(iq, fs=fs, audio_cutoff=3_000)
        corr = self._aligned_correlation(message, demod)
        self.assertGreater(corr, 0.9)

    def test_load_complex_capture_reads_npz_sample_rate(self):
        iq = np.array([1 + 1j, 2 - 1j, -0.5 + 0.25j], dtype=np.complex128)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "capture.npz"
            np.savez(path, iq=iq, sample_rate=240_000)
            fs, loaded = rf_utils.load_complex_capture(path)
        self.assertEqual(fs, 240_000)
        np.testing.assert_allclose(loaded, iq)

    def test_plot_helpers_return_axes(self):
        fs = 8_000
        _, tone = rf_utils.generate_tone(freq=500, duration=0.2, fs=fs)
        waveform_ax = rf_utils.plot_waveform(tone, fs=fs, title="Wave")
        spectrum_ax = rf_utils.plot_spectrum(tone, fs=fs, title="Spec")
        spectrogram_ax, mesh = rf_utils.plot_spectrogram(tone, fs=fs, title="Gram")
        self.assertEqual(waveform_ax.get_title(), "Wave")
        self.assertEqual(spectrum_ax.get_title(), "Spec")
        self.assertEqual(spectrogram_ax.get_title(), "Gram")
        self.assertIsNotNone(mesh)

    def test_widget_helpers_create_expected_controls(self):
        slider = rf_utils.float_slider(
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.5,
            description="Test",
        )
        int_slider = rf_utils.int_slider(
            min_value=1,
            max_value=5,
            step=1,
            value=3,
            description="Count",
        )
        dropdown = rf_utils.dropdown(
            options=["hann", "hamming"],
            value="hann",
            description="Window",
        )
        output = rf_utils.audio_output_widget()

        self.assertEqual(slider.description, "Test")
        self.assertEqual(int_slider.value, 3)
        self.assertEqual(dropdown.value, "hann")
        self.assertEqual(output.__class__.__name__, "Output")


if __name__ == "__main__":
    unittest.main()
