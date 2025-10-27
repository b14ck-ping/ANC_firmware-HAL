#!/usr/bin/env python3
"""
linear_spectrum.py

Compute averaged spectrum (Welch) and plot it in LINEAR scale (not dB).

Outputs:
  - PNG plot per channel: chan{n}_avg_spectrum_linear.png
  - Numpy files with freq and PSD (or ASD): chan{n}_psd_freq.npy, chan{n}_psd.npy

Usage:
  python linear_spectrum.py --files chan0.npy chan1.npy --fs 16000 --mode psd --yscale linear --xscale log --out outdir

Options:
  --mode {psd,asd}    : 'psd' outputs Power Spectral Density (units: unit^2/Hz)
                        'asd' outputs Amplitude Spectral Density (units: unit/√Hz)
  --is-db             : input arrays are in dB SPL (will convert back to Pascals)
  --nperseg INT       : window length for Welch (default 1024)
  --noverlap INT      : overlap (default nperseg//2)
  --xscale {log,lin}  : frequency axis scale (default: log)
  --yscale {linear,log}: y axis scale (default linear)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

p0 = 20e-6

def load_array(path, is_db=False):
    a = np.load(path)
    if is_db:
        # assume input is dB SPL -> convert to Pascals
        a = p0 * (10.0 ** (a / 20.0))
    return a.astype(np.float64)

def compute_welch(x, fs, nperseg=1024, noverlap=None, window='hann'):
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = signal.welch(x, fs=fs, window=window, nperseg=nperseg,
                          noverlap=noverlap, detrend='constant', scaling='density', axis=-1)
    return f, Pxx

def save_and_plot(f, data, channel, outdir, mode='psd', xscale='log', yscale='linear', xlabel='Frequency (Hz)'):
    os.makedirs(outdir, exist_ok=True)
    if mode == 'psd':
        ylabel = 'PSD (linear units²/Hz)'
        fname = f'chan{channel}_psd_linear.png'
        npyname = f'chan{channel}_psd.npy'
    else:
        ylabel = 'ASD (linear units/√Hz)'
        fname = f'chan{channel}_asd_linear.png'
        npyname = f'chan{channel}_asd.npy'

    plt.figure(figsize=(8,4))
    if xscale == 'log':
        plt.semilogx(f, data)
    else:
        plt.plot(f, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Channel {channel} averaged {mode.upper()} (linear scale)')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    if yscale == 'log':
        plt.yscale('log')
    plt.tight_layout()
    outpng = os.path.join(outdir, fname)
    outnpy = os.path.join(outdir, npyname)
    np.save(os.path.join(outdir, f'chan{channel}_psd_freq.npy'), f)
    np.save(outnpy, data)
    plt.savefig(outpng, dpi=200)
    plt.close()
    print(f"Saved plot: {outpng}")
    print(f"Saved arrays: {outnpy}, freq saved as chan{channel}_psd_freq.npy")

def compute_and_save(files, fs, outdir, mode='psd', nperseg=1024, noverlap=None, is_db=False, xscale='log', yscale='linear'):
    arrays = []
    for path in files:
        a = load_array(path, is_db=is_db)
        arrays.append(a)

    for idx, arr in enumerate(arrays):
        if arr.size == 0:
            print(f"Channel {idx} empty, skipping.")
            continue
        f, Pxx = compute_welch(arr, fs, nperseg=nperseg, noverlap=noverlap)
        if mode == 'psd':
            data = Pxx  # unit^2/Hz
        else:
            data = np.sqrt(Pxx)  # unit/√Hz
        save_and_plot(f, data, idx, outdir, mode=mode, xscale=xscale, yscale=yscale)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--files', '-f', nargs='+', required=True, help='Input .npy files (one per channel)')
    p.add_argument('--fs', type=float, default=16000.0, help='Sampling rate (Hz)')
    p.add_argument('--out', '-o', default='linear_spectrum_out', help='Output directory')
    p.add_argument('--mode', choices=['psd','asd'], default='psd', help='psd or asd (amplitude spectral density)')
    p.add_argument('--is-db', action='store_true', help='Input arrays are in dB SPL (convert back to Pa)')
    p.add_argument('--nperseg', type=int, default=1024, help='Welch window length')
    p.add_argument('--noverlap', type=int, default=None, help='Welch overlap (default nperseg//2)')
    p.add_argument('--xscale', choices=['log','lin'], default='log', help='X axis scale (frequency)')
    p.add_argument('--yscale', choices=['linear','log'], default='linear', help='Y axis scale')
    return p.parse_args()

def main():
    args = parse_args()
    compute_and_save(args.files, fs=args.fs, outdir=args.out, mode=args.mode, nperseg=args.nperseg, noverlap=args.noverlap, is_db=args.is_db, xscale=args.xscale, yscale=args.yscale)

if __name__ == '__main__':
    main()
