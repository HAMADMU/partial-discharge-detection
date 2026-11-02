"""Turn each raw signal into a small set of features describing its discharge activity.

Partial discharge shows up as short, sharp pulses on top of the 50 Hz waveform.
A high-pass filter removes the 50 Hz component so the pulses stand out, then the
features count and size those pulses relative to the signal's own noise level.
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from load_data import SAMPLING_RATE_HZ

HIGHPASS_CUTOFF_HZ = 10_000   # well above 50 Hz, well below the PD pulse content
PEAK_THRESHOLD_SIGMAS = 5     # a pulse must be this many noise-sigmas tall
MIN_PEAK_SPACING = 50         # samples (1.25 microseconds) between counted peaks


def highpass(signal, sampling_rate=SAMPLING_RATE_HZ):
    sos = butter(5, HIGHPASS_CUTOFF_HZ, btype="highpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, signal.astype(np.float64))


def robust_noise_sigma(x):
    """Noise level from the median absolute value, which large pulses barely affect."""
    return np.median(np.abs(x)) / 0.6745 + 1e-9


def signal_features(signal, sampling_rate=SAMPLING_RATE_HZ):
    """Return a dict of features for one signal."""
    filtered = highpass(signal, sampling_rate)
    sigma = robust_noise_sigma(filtered)
    magnitude = np.abs(filtered)

    peaks, props = find_peaks(magnitude, height=PEAK_THRESHOLD_SIGMAS * sigma,
                              distance=MIN_PEAK_SPACING)
    heights = props["peak_heights"]
    n_positive = int(np.sum(filtered[peaks] > 0))

    return {
        "noise_sigma": sigma,
        "filtered_std": float(np.std(filtered)),
        "max_abs": float(magnitude.max()),
        "p99_abs": float(np.percentile(magnitude, 99)),
        "n_peaks": int(len(peaks)),
        "n_positive_peaks": n_positive,
        "n_negative_peaks": int(len(peaks) - n_positive),
        "mean_peak_height": float(heights.mean()) if len(heights) else 0.0,
        "max_peak_over_noise": float(heights.max() / sigma) if len(heights) else 0.0,
    }
