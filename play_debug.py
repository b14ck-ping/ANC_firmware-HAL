#!/usr/bin/env python3
"""
play_debug.py

Debug playback: распечатает инфу о файле (.npy/.wav/.csv), проверит форму, dtype и рассчитает ожидаемую длительность.
Потом безопасно воспроизведёт (blocking) с возможностью указать частоту --fs и канал.

Usage examples:
  python play_debug.py --file out/chan0.npy --fs 16000 --normalize
  python play_debug.py --file recording.wav
"""

import argparse, os, sys
import numpy as np
import sounddevice as sd
import soundfile as sf

def load_and_inspect(path, channel=0, fs_arg=None):
    ext = os.path.splitext(path)[1].lower()
    info = {}
    if ext == '.npy':
        arr = np.load(path, allow_pickle=False)
        info['orig_shape'] = arr.shape
        if arr.ndim == 1:
            samples = arr.astype(np.float32)
        elif arr.ndim == 2:
            # Try to infer layout:
            # if shape[0] is small (<=8) -> treat as (channels, samples)
            if arr.shape[0] <= 8 and arr.shape[0] != arr.shape[1]:
                # (channels, samples)
                if channel >= arr.shape[0]:
                    raise ValueError("Requested channel out of range")
                samples = arr[channel, :].astype(np.float32)
                info['assumed_layout'] = '(channels, samples)'
            else:
                # treat as (samples, channels)
                if channel >= arr.shape[1]:
                    raise ValueError("Requested channel out of range")
                samples = arr[:, channel].astype(np.float32)
                info['assumed_layout'] = '(samples, channels)'
        else:
            raise ValueError("Unsupported numpy array shape: ndim > 2")
        info['file_type'] = 'npy'
        info['fs'] = fs_arg
    elif ext in ('.wav', '.flac', '.aiff', '.aif', '.ogg'):
        data, fs = sf.read(path, always_2d=True)
        info['file_type'] = 'wav'
        info['orig_shape'] = data.shape
        info['fs'] = fs
        if channel >= data.shape[1]:
            raise ValueError("Requested channel out of range for audio file")
        samples = data[:, channel].astype(np.float32)
    elif ext in ('.csv', '.txt'):
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim == 2 and arr.shape[1] > 1:
            samples = arr[:, 1].astype(np.float32)
            info['orig_shape'] = arr.shape
        else:
            samples = arr.flatten().astype(np.float32)
            info['orig_shape'] = samples.shape
        info['file_type'] = 'csv'
        info['fs'] = fs_arg
    else:
        raise ValueError("Unsupported extension: " + ext)

    if info.get('fs', None) is None:
        raise ValueError("Sampling rate unknown: provide --fs for .npy/.csv input")

    info['samples_len'] = samples.shape[0]
    info['dtype'] = samples.dtype
    info['min'] = float(np.min(samples)) if samples.size>0 else 0.0
    info['max'] = float(np.max(samples)) if samples.size>0 else 0.0
    info['mean_abs'] = float(np.mean(np.abs(samples))) if samples.size>0 else 0.0
    info['duration_s'] = samples.shape[0] / float(info['fs']) if info['fs'] else None
    return samples, info

def normalize(x, peak=0.99):
    m = np.max(np.abs(x))
    if m <= 0: return x
    return x * (peak / m)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file','-f', required=True)
    p.add_argument('--fs', type=float, default=None, help='Sampling rate if input is .npy or .csv')
    p.add_argument('--channel','-c', type=int, default=0)
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--gain-db', type=float, default=0.0)
    p.add_argument('--repeat', type=int, default=1, help='Repeat playback N times (debug)')
    args = p.parse_args()

    try:
        samples, info = load_and_inspect(args.file, channel=args.channel, fs_arg=args.fs)
    except Exception as e:
        print("Load error:", e, file=sys.stderr)
        sys.exit(2)

    print("=== FILE INFO ===")
    for k,v in info.items():
        print(f"{k}: {v}")
    print("=================")

    # quick peek of first / last samples
    n = samples.shape[0]
    print("first 10 samples:", samples[:10].tolist())
    print("last 10 samples:", samples[-10:].tolist())
    print("non-zero count:", int(np.count_nonzero(samples)))
    # stats per chunk
    window = samples[:min(1000,n)]
    print("first 1000 window mean abs:", float(np.mean(np.abs(window))) )

    fs = int(info['fs'])
    # apply gain if requested
    if abs(args.gain_db) > 1e-6:
        factor = 10.0 ** (args.gain_db/20.0)
        samples = samples * factor
        print(f"Applied gain {args.gain_db} dB (factor {factor:.3g})")

    if args.normalize:
        samples = normalize(samples)
        print("Normalized to peak 0.99")

    peak = float(np.max(np.abs(samples)))
    print(f"After preprocessing: samples={samples.shape[0]} duration={samples.shape[0]/fs:.3f}s peak={peak:.6f}")

    if samples.shape[0] == 0:
        print("Empty sample array, abort.")
        return

    # ensure float32 and in [-1..1] safe range
    samples = samples.astype(np.float32)
    if peak > 4.0:
        print("Warning: big peak >4.0, clipping very likely when playing.", file=sys.stderr)

    try:
        sd.default.samplerate = fs
        sd.default.channels = 1
        for i in range(max(1,args.repeat)):
            print(f"Playback {i+1}/{args.repeat} ...")
            sd.play(samples, fs, blocking=True)
        print("Playback finished.")
    except Exception as e:
        print("Playback failed:", e, file=sys.stderr)
        sys.exit(3)

if __name__ == '__main__':
    main()
