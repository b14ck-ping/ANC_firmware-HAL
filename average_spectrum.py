#!/usr/bin/env python3
"""
average_spectrum.py

Compute averaged spectrum (Welch) for one or more channels,
plot and save results.

Usage example:
    python average_spectrum.py --files out/chan0.npy out/chan1.npy --fs 16000 --is-db

Or from Python API:
    arrays = {0: np.load('chan0.npy'), 1: np.load('chan1.npy')}
    compute_and_plot_average_spectrum(arrays, fs=16000, is_db_input=True)

Outputs:
 - PNG plots per channel: chan{n}_avg_spectrum.png
 - Numpy files with freq and PSD_dB: chan{n}_psd_freq.npy, chan{n}_psd_db.npy
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

p0 = 20e-6  # reference pressure (Pa) for dB SPL

def compute_avg_psd(x, fs, nperseg=1024, noverlap=None, window='hann', is_db_input=False):
    """
    Compute averaged PSD (Welch). 
    - x: 1D numpy array (time series)
    - fs: sampling frequency (Hz)
    - is_db_input: if True, x holds dB SPL values -> converted to Pa before PSD
    Returns: f (Hz), Pxx (PSD, Pa^2/Hz)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    # if input is dB SPL, convert back to Pascals
    if is_db_input:
        # x is dB SPL (20*log10(p/p0)), convert: p = p0 * 10^(dB/20)
        p = p0 * (10.0 ** (x / 20.0))
    else:
        p = x.astype(np.float64)

    # Use scipy.signal.welch for averaged PSD (density)
    f, Pxx = signal.welch(p, fs=fs, window=window, nperseg=nperseg,
                          noverlap=noverlap, detrend='constant', scaling='density', axis=-1)
    return f, Pxx

def psd_to_db_spl(Pxx, ref_p0=p0):
    """
    Convert PSD (Pa^2/Hz) to dB SPL/Hz: 10*log10(Pxx / p0^2)
    (This is dB relative to p0^2; commonly shown as dB re 20 µPa^2/Hz.)
    """
    # avoid log of zero
    Pxx_safe = np.maximum(Pxx, 1e-20)
    return 10.0 * np.log10(Pxx_safe / (ref_p0 ** 2))

def plot_and_save(f, Pxx_db, channel, outdir, xlabel='Frequency (Hz)', ylabel='PSD (dB re 20µPa^2/Hz)'):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.semilogx(f, Pxx_db)          # log-frequency axis often nicer
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Channel {channel} — averaged spectrum (Welch)')
    plt.xlim(10, f.max())
    plt.tight_layout()
    png_path = os.path.join(outdir, f'chan{channel}_avg_spectrum.png')
    npy_freq = os.path.join(outdir, f'chan{channel}_psd_freq.npy')
    npy_db = os.path.join(outdir, f'chan{channel}_psd_db.npy')
    plt.savefig(png_path, dpi=200)
    plt.close()
    # save numeric
    np.save(npy_freq, f)
    np.save(npy_db, Pxx_db)
    print(f"Saved plot: {png_path}")
    print(f"Saved arrays: {npy_freq}, {npy_db}")

def compute_and_plot_average_spectrum(arrays, fs, outdir='out', nperseg=1024, noverlap=None, is_db_input=False):
    """
    arrays: dict {channel_id: 1D numpy array}
    fs: sampling freq
    """
    for ch, a in arrays.items():
        if len(a) == 0:
            print(f"Channel {ch} is empty, skipping.")
            continue
        f, Pxx = compute_avg_psd(a, fs, nperseg=nperseg, noverlap=noverlap, is_db_input=is_db_input)
        Pxx_db = psd_to_db_spl(Pxx)
        plot_and_save(f, Pxx_db, ch, outdir)
    print("Done.")


def parse_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', nargs='+', required=True,
                        help='Input .npy files, in order (e.g. chan0.npy chan1.npy) or other numpy-readable arrays')
    parser.add_argument('--fs', type=float, default=16000.0, help='Sampling frequency in Hz')
    parser.add_argument('--out', '-o', default='spectrum_out', help='Output directory')
    parser.add_argument('--nperseg', type=int, default=1024, help='Window length for Welch')
    parser.add_argument('--noverlap', type=int, default=None, help='Overlap for Welch (None => nperseg/2)')
    parser.add_argument('--is-db', action='store_true', help='Input arrays are in dB SPL (convert back to Pa)')
    args = parser.parse_args()

    arrays = {}
    for i, fpath in enumerate(args.files):
        data = np.load(fpath)
        arrays[i] = data
        print(f"Loaded {fpath}: {data.shape[0]} samples")

    compute_and_plot_average_spectrum(arrays, fs=args.fs, outdir=args.out, nperseg=args.nperseg, noverlap=args.noverlap, is_db_input=args.is_db)


if __name__ == '__main__':
    parse_and_run()
