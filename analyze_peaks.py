#!/usr/bin/env python3
"""
analyze_peaks.py

Detect and analyze transient peaks in a recorded signal (ADC codes or Pascals).
Generates plots, CSV with peak list, short WAV snippets around peaks, PSD and spectrogram.

Usage examples:
  python analyze_peaks.py --file out1/chan0_adc.npy --mode adc --fs 32000 --out diag_out --short-seconds 3
  python analyze_peaks.py --file out1/chan0_pa.npy --mode pa --fs 32000 --out diag_out

Outputs (in --out directory):
  - wave_peaks.png (full signal with peaks marked)
  - psd.png, spectrogram.png
  - peaks.csv (index, time_s, value, prominence, width_samples)
  - peaks_snips/peak_000.png and .wav (zoom and audio for top peaks)
  - cleaned.wav (signal after conservative median-based spike removal)
  - report.txt (summary statistics)
"""
import argparse
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import soundfile as sf
from scipy.ndimage import median_filter

P0 = 20e-6

def human(x):
    try:
        if abs(x) >= 1:
            return f"{x:.3f}"
        else:
            return f"{x:.3e}"
    except:
        return str(x)

def analyze(arr, fs, mode, outdir, short_seconds=None):
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.file))[0]
    N = len(arr)
    dur = N / fs
    t = np.arange(N) / fs

    # Basic stats
    mean = float(np.mean(arr))
    rms = float(np.sqrt(np.mean((arr - 0.0)**2)))
    peak = float(np.max(np.abs(arr)))
    p2p = float(np.ptp(arr))
    crest = peak / (rms + 1e-30)
    # If Pa, compute RMS dB SPL
    dbspl = None
    if mode == 'pa':
        dbspl = 20.0 * math.log10(max(rms,1e-20) / P0)

    # Save summary
    with open(os.path.join(outdir, 'report.txt'), 'w') as f:
        f.write(f"file: {args.file}\n")
        f.write(f"mode: {mode}\n")
        f.write(f"samples: {N}\n")
        f.write(f"fs: {fs}\n")
        f.write(f"duration_s: {dur:.3f}\n")
        f.write(f"mean: {mean:.6e}\n")
        f.write(f"rms: {rms:.6e}\n")
        f.write(f"peak: {peak:.6e}\n")
        f.write(f"p2p: {p2p:.6e}\n")
        f.write(f"crest: {crest:.2f}\n")
        if dbspl is not None:
            f.write(f"est RMS dB SPL: {dbspl:.2f} dB\n")
    print("[i] summary written to report.txt")

    # Full time plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(t, arr, linewidth=0.5)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Pa' if mode == 'pa' else 'ADC')
    ax.set_title(f"{basename} time series")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'waveform.png'), dpi=150)
    plt.close()

    # PSD (Welch)
    f, Pxx = signal.welch(arr, fs=fs, nperseg=min(4096, N))
    plt.figure(figsize=(10,4))
    plt.semilogy(f, Pxx)
    plt.xlabel('Hz'); plt.ylabel('PSD')
    plt.title('Welch PSD')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'psd.png'), dpi=150)
    plt.close()

    # Spectrogram (short)
    plt.figure(figsize=(10,4))
    nfft = 1024
    nover = nfft//2
    f_s, t_s, Sxx = signal.spectrogram(arr, fs=fs, nperseg=nfft, noverlap=nover, scaling='density', mode='magnitude')
    plt.pcolormesh(t_s, f_s, 20*np.log10(np.maximum(Sxx,1e-20)), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.ylim(0, min(fs/2, 16000))
    plt.title('Spectrogram (dB)')
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'spectrogram.png'), dpi=150)
    plt.close()

    # Peak detection - adaptive thresholds
    # We'll detect both positive and negative transient peaks by absolute value
    abs_arr = np.abs(arr)
    # baseline noise estimate: median absolute value
    med = np.median(abs_arr)
    mad = np.median(np.abs(abs_arr - med)) + 1e-30
    # heuristic: detect peaks above median + k * MAD
    k = 8.0
    height_thr = med + k * mad
    # minimal distance between peaks (in samples) to avoid duplicates
    min_distance = int(0.001 * fs)  # 1 ms by default
    # find peaks on absolute waveform
    peaks, props = find_peaks(abs_arr, height=height_thr, distance=min_distance, prominence=mad*4)
    prominences = props.get('prominences', None)
    widths = signal.peak_widths(abs_arr, peaks, rel_height=0.5)[0] if peaks.size>0 else np.array([])

    # If no peaks found, relax threshold
    if peaks.size == 0:
        height_thr = med + 4.0 * mad
        peaks, props = find_peaks(abs_arr, height=height_thr, distance=min_distance//2, prominence=mad*2)
        prominences = props.get('prominences', None)
        widths = signal.peak_widths(abs_arr, peaks, rel_height=0.5)[0] if peaks.size>0 else np.array([])

    print(f"[i] detected {len(peaks)} peaks (threshold {height_thr:.3e})")

    # Save peaks CSV
    import csv
    csv_path = os.path.join(outdir, 'peaks.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['idx', 'time_s', 'value', 'abs_value', 'prominence', 'width_samples'])
        for i, idx in enumerate(peaks):
            prom = prominences[i] if prominences is not None and i < len(prominences) else ''
            width = int(widths[i]) if i < len(widths) else ''
            w.writerow([int(idx), float(idx)/fs, float(arr[idx]), float(abs_arr[idx]), prom, width])
    print(f"[i] peaks list saved to {csv_path}")

    # Plot waveform with peaks marked (full)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(t, arr, linewidth=0.4)
    if peaks.size > 0:
        ax.plot(peaks/fs, arr[peaks], 'r.', markersize=6)
    ax.set_xlabel('s'); ax.set_ylabel('Pa' if mode=='pa' else 'ADC')
    ax.set_title('Waveform with detected peaks')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'wave_peaks.png'), dpi=150)
    plt.close()

    # Create directory for snippets
    snip_dir = os.path.join(outdir, 'peaks_snips')
    os.makedirs(snip_dir, exist_ok=True)

    # Save top K peaks by prominence (or amplitude)
    K = 20
    if peaks.size > 0:
        # sort by prominence or abs value
        order = np.argsort(-abs_arr[peaks])
        top_idx = peaks[order][:K]
    else:
        top_idx = np.array([], dtype=int)

    for rank, center in enumerate(top_idx):
        half_ms = 50  # milliseconds on each side
        half_samples = int((half_ms/1000.0) * fs)
        a0 = max(0, center - half_samples)
        a1 = min(N, center + half_samples)
        snippet = arr[a0:a1]
        tm = np.arange(a0, a1)/fs
        # save zoom plot
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(tm, snippet, linewidth=0.6)
        ax.axvline(center/fs, color='r', linestyle='--')
        ax.set_title(f'Peak #{rank} idx={center} t={center/fs:.4f}s val={arr[center]:.3e}')
        ax.grid(True)
        plt.tight_layout()
        png = os.path.join(snip_dir, f'peak_{rank:03d}.png')
        plt.savefig(png, dpi=150)
        plt.close()
        # save WAV for snippet (normalize by peak of snippet to safe level)
        # convert numeric range to float -1..1 safely: scale by snippet peak
        snip_peak = np.max(np.abs(snippet)) + 1e-30
        audio = snippet.astype(np.float32) / snip_peak * 0.6
        wavpath = os.path.join(snip_dir, f'peak_{rank:03d}.wav')
        sf.write(wavpath, audio, int(fs))
    print(f"[i] saved top {len(top_idx)} snippet PNGs/WAVs into {snip_dir}")

    # Quick cleaning attempt: median filter on absolute values to remove spikes
    # We'll do temporal median filtering on raw arr: replace samples exceeding threshold with median around window
    cleaned = arr.copy()
    # mark spikes by comparing to local median
    win = int(0.002 * fs)  # 2 ms window
    if win < 3: win = 3
    med_local = signal.medfilt(arr, kernel_size=win if win%2==1 else win+1)
    spike_mask = np.abs(arr - med_local) > (5 * mad + 1e-30)  # heuristic
    cleaned[spike_mask] = med_local[spike_mask]
    # Save cleaned WAV (peak-normalized to safe level)
    if mode == 'pa':
        # scale to audio-friendly range: normalize by peak
        cp = np.max(np.abs(cleaned)) + 1e-30
        audio_clean = (cleaned / cp * 0.8).astype(np.float32)
    else:
        cp = np.max(np.abs(cleaned)) + 1e-30
        audio_clean = (cleaned / cp * 0.8).astype(np.float32)
    sf.write(os.path.join(outdir, 'cleaned.wav'), audio_clean, int(fs))
    print(f"[i] saved cleaned.wav (median spike-suppressed)")

    # Save some diagnostics numbers to report
    with open(os.path.join(outdir, 'report.txt'), 'a') as f:
        f.write(f"detected_peaks: {len(peaks)}\n")
        f.write(f"peak_threshold: {height_thr:.6e}\n")
        f.write(f"median_abs: {med:.6e}, mad: {mad:.6e}\n")
    print("[i] done analysis.")

if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="analyze_peaks.py")
    p.add_argument('--file', required=True, help='input .npy file (adc or pa)')
    p.add_argument('--mode', choices=['adc','pa'], default='adc', help='interpretation of input')
    p.add_argument('--fs', type=float, required=True, help='sample rate (Hz)')
    p.add_argument('--out', default='diag_out', help='output directory')
    p.add_argument('--short-seconds', type=float, default=None, help='save a short prefix of this many seconds as sample_short.npy for sharing')
    args = p.parse_args()

    x = np.load(args.file)
    if args.short_seconds:
        Nsamp = int(min(x.size, int(args.short_seconds * args.fs)))
        np.save(os.path.join(args.out, 'sample_short.npy'), x[:Nsamp])
        print(f"[+] saved short sample {Nsamp} samples -> {os.path.join(args.out, 'sample_short.npy')}")
    analyze(x, args.fs, args.mode, args.out, short_seconds=args.short_seconds)
