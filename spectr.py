#!/usr/bin/env python3
"""
uart_capture_and_spectrum.py

Read MCU telemetry packets from serial, save per-channel dB time series and compute averaged spectra.

Packet format:
  [0xAA,0x55]         - preamble (2 bytes)
  [ver: uint8]        - protocol version (1 byte)
  [seq: uint16 LE]    - packet sequence number (2 bytes)
  [len: uint8]        - payload length in bytes (1 byte)  -> expected 3
  [payload: len bytes]- payload (id:uint8 + q:int16 LE)
  [crc16: uint16 LE]  - CRC16 over ver..payload (not preamble)

Payload: id = 0 or 1, q = round(db * 100) -> db = q / 100.0

Usage:
  python uart_capture_and_spectrum.py --port COM5 --baud 921600 --duration 10 --out outdir --nperseg 1024 --is-db

Outputs:
  outdir/chan0.npy, outdir/chan1.npy
  outdir/chan0_avg_spectrum.png (dB)
  outdir/chan0_psd_freq.npy, outdir/chan0_psd_db.npy
  (same for channel 1)
  outdir/summary.txt
"""

import argparse
import serial
import time
import os
import sys
import struct
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Reference pressure for dB SPL conversions when needed
P0 = 20e-6

# АЦП и преобразование
VREF = 3.3  # опорное напряжение АЦП
ADC_MAX = 4095.0  # максимальный код 12-бит АЦП
V_BIAS = 1.5  # DC смещение микрофона MAX9814
MIC_GAIN_DB = 40.0  # коэффициент усиления MAX9814
MIC_SENSITIVITY_DBV = -44.0  # чувствительность капсюля микрофона, dBV/Pa

# Расчет промежуточных констант
MIC_GAIN_LINEAR = 10.0 ** (MIC_GAIN_DB / 20.0)
MIC_SENS_V_PA = 10.0 ** (MIC_SENSITIVITY_DBV / 20.0)

def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init & 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ 0x1021
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF

PRE0 = 0xAA
PRE1 = 0x55
PKT_LEN = 9  # total bytes per packet

def adc_to_pa(adc_code):
    """Преобразование кода АЦП в паскали"""
    # АЦП в вольты
    v_sig = (adc_code / ADC_MAX) * VREF
    # Убираем смещение
    v_ac = v_sig - V_BIAS
    # Вольты в паскали через коэффициент усиления и чувствительность
    pa = v_ac / (MIC_GAIN_LINEAR * MIC_SENS_V_PA)
    return pa

def parse_chunk_buffer(buf: bytearray, ch_data: dict, stats: dict):
    """
    Parse fixed-size 9-byte packets from buf:
      [0]=PRE0, [1]=PRE1, [2]=id, [3:7]=cnt (uint32 LE), [7:9]=q (int16 LE)
    Consumes parsed bytes from buf.

    Tracks:
      - stats['base_cnt'] : first observed absolute cnt (uint32)
      - stats['duplicates'] : count of duplicate packets (same id & cnt as last for that channel)
      - stats['gaps_total'] : total missing sample-count across all channels
      - stats['gaps_list'] : list of gap events: dicts {ch, from_rel, to_rel, missing}
    Sample numbering in reports is relative to first received sample (base = 0).
    """
    i = 0
    parsed_any = False
    # per-channel last absolute cnt
    last_cnt = stats.setdefault('_last_cnt_per_ch', {})
    base_cnt = stats.get('base_cnt', None)

    while i + PKT_LEN <= len(buf):
        # find preamble
        if buf[i] != PRE0 or buf[i+1] != PRE1:
            i += 1
            continue
        # full packet available
        pkt = buf[i:i+PKT_LEN]
        ch_id = pkt[2]
        cnt = int.from_bytes(pkt[3:7], 'little', signed=False)  # uint32
        q = int.from_bytes(pkt[7:9], 'little', signed=True)     # keep signed as before
        # initialize base_cnt on first received sample
        if base_cnt is None:
            stats['base_cnt'] = cnt
            base_cnt = cnt
        # compute relative sample index (uint32 wrap safe)
        rel = (cnt - base_cnt) & 0xFFFFFFFF

        # detect duplicate vs gap per channel
        prev = last_cnt.get(ch_id, None)
        if prev is None:
            # first packet for this channel: just record
            pass
        else:
            # calculate unsigned difference (wrap-safe)
            diff = (cnt - prev) & 0xFFFFFFFF
            if diff == 0:
                # duplicate packet for same channel and same cnt
                stats['duplicates'] = stats.get('duplicates', 0) + 1
                dlist = stats.setdefault('duplicates_list', [])
                dlist.append({'ch': ch_id, 'cnt': cnt, 'rel': rel})
            elif diff > 1:
                # gap detected: missing (diff - 1) samples between prev and cnt
                missing = diff - 1
                stats['gaps_total'] = stats.get('gaps_total', 0) + missing
                glist = stats.setdefault('gaps_list', [])
                # report missing range in relative numbering
                from_rel = ((prev - base_cnt) & 0xFFFFFFFF) + 1
                to_rel = rel - 1
                glist.append({'ch': ch_id, 'from_rel': int(from_rel), 'to_rel': int(to_rel), 'missing': int(missing)})
        # update last cnt for channel
        last_cnt[ch_id] = cnt

       # Преобразуем код АЦП в паскали
        pa = adc_to_pa(q)
        ch_data.setdefault(ch_id, []).append(pa)
        
        stats['packets_parsed'] = stats.get('packets_parsed', 0) + 1
        parsed_any = True
        i += PKT_LEN

    if i:
        del buf[:i]
    return parsed_any

def compute_and_save_welch(ch_arrays: dict, fs: float, outdir: str, nperseg: int=1024, 
                          noverlap=None, is_db_input=False, output_units='pa'):
    """
    Compute averaged spectrum using Welch's method.
    Input: raw ADC values
    Output: PSD in Pa²/Hz or dB re 20µPa²/Hz
    """
    os.makedirs(outdir, exist_ok=True)
    if noverlap is None:
        noverlap = nperseg // 2

    for ch, arr in ch_arrays.items():
        if len(arr) < 4:
            print(f"[i] channel {ch} has too few samples ({len(arr)}), skipping spectrum")
            continue

        # Input data is already in Pascals from parse_chunk_buffer
        x = np.asarray(arr, dtype=np.float64)
        
        # Compute PSD using Welch's method
        f, Pxx = signal.welch(x, fs=fs, window='hann', nperseg=nperseg,
                            noverlap=noverlap, detrend='constant',
                            scaling='density', axis=-1)
        
        # Convert to dB SPL if requested
        if output_units == 'db':
            Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20) / (P0**2))
            data_to_plot = Pxx_db
            ylab = 'PSD (dB re 20µPa²/Hz)'
            suffix = 'psd_db'
        else:
            data_to_plot = Pxx
            ylab = 'PSD (Pa²/Hz)'
            suffix = 'psd_pa'

        # Save raw data
        np.save(os.path.join(outdir, f'chan{ch}_psd_freq.npy'), f)
        np.save(os.path.join(outdir, f'chan{ch}_{suffix}.npy'), data_to_plot)

        # Plot
        plt.figure(figsize=(8,4))
        plt.semilogx(f, data_to_plot)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(ylab)
        plt.title(f'Channel {ch} Power Spectral Density')
        plt.xlim(10, f.max())
        
        # Add some statistical info
        if output_units == 'db':
            rms_pa = np.sqrt(np.mean(x**2))
            db_spl = 20 * np.log10(rms_pa / P0)
            plt.text(0.02, 0.98, f'RMS: {db_spl:.1f} dB SPL', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top')

        plt.tight_layout()
        png = os.path.join(outdir, f'chan{ch}_spectrum_{suffix}.png')
        plt.savefig(png, dpi=200)
        plt.close()
        
        print(f"[+] saved {png} and PSD arrays (npy)")

def compute_and_save_linear(ch_arrays: dict, fs: float, outdir: str, nperseg: int=1024, noverlap=None, is_db_input=False, mode='psd'):
    """
    Save linear PSD or ASD (mode='psd' or 'asd').
    PSD units: unit^2/Hz (Pa^2/Hz if input Pa)
    ASD units: unit/√Hz
    """
    os.makedirs(outdir, exist_ok=True)
    if noverlap is None:
        noverlap = nperseg // 2
    for ch, arr in ch_arrays.items():
        if len(arr) < 4:
            continue
        x = np.asarray(arr, dtype=np.float64)
        if is_db_input:
            p = P0 * (10.0 ** (x / 20.0))
        else:
            p = x
        f, Pxx = signal.welch(p, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, detrend='constant', scaling='density', axis=-1)
        if mode == 'psd':
            data = Pxx
            ylab = 'PSD (unit²/Hz)'
            suffix = 'psd_linear'
        else:
            data = np.sqrt(Pxx)
            ylab = 'ASD (unit/√Hz)'
            suffix = 'asd_linear'
        png = os.path.join(outdir, f'chan{ch}_{suffix}.png')
        np.save(os.path.join(outdir, f'chan{ch}_{suffix}_freq.npy'), f)
        np.save(os.path.join(outdir, f'chan{ch}_{suffix}.npy'), data)
        plt.figure(figsize=(8,4))
        plt.semilogx(f, data)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(ylab)
        plt.title(f'Channel {ch} averaged {mode} (linear)')
        plt.xlim(10, f.max())
        plt.tight_layout()
        plt.savefig(png, dpi=200)
        plt.close()
        print(f"[+] saved {png} and arrays")

def save_time_series(outdir: str, ch_data: dict):
    os.makedirs(outdir, exist_ok=True)
    for ch, arr in ch_data.items():
        path = os.path.join(outdir, f'chan{ch}.npy')
        np.save(path, np.asarray(arr, dtype=np.float32))
        print(f"[+] saved channel {ch} {len(arr)} samples -> {path}")

def write_summary(outdir: str, stats: dict, ch_data: dict):
    path = os.path.join(outdir, "summary.txt")
    with open(path, "w") as f:
        # write standard stats (keep compatibility)
        for k in ('packets_parsed','bad_crc','malformed','seq_lost','last_seq'):
            if k in stats:
                f.write(f"{k}={stats[k]}\n")
        # write added counters
        f.write(f"duplicates={stats.get('duplicates',0)}\n")
        f.write(f"gaps_total={stats.get('gaps_total',0)}\n")
        # optionally dump gaps_list and duplicates_list (compact)
        glist = stats.get('gaps_list', [])
        for idx, g in enumerate(glist):
            f.write(f"gap_{idx}=ch{g['ch']} from_rel={g['from_rel']} to_rel={g['to_rel']} missing={g['missing']}\n")
        dlist = stats.get('duplicates_list', [])
        for idx, d in enumerate(dlist):
            f.write(f"dup_{idx}=ch{d['ch']} cnt_rel={(d['cnt'] - stats.get('base_cnt',0)) & 0xFFFFFFFF}\n")

        f.write("\n")
        for ch in sorted(ch_data.keys()):
            f.write(f"chan{ch}_samples={len(ch_data[ch])}\n")
    print(f"[+] saved summary -> {path}")

def run_capture(port: str, baud: int, duration: float, outdir: str, fs_db: float, nperseg: int, is_db_input: bool, compute_linear: bool, linear_mode: str, output_units: str):
    ser = serial.Serial(port, baud, timeout=0.1)
    buf = bytearray()
    ch_data = {}
    stats = {
        'packets_parsed': 0,
        'bad_crc': 0,
        'malformed': 0,
        'seq_lost': 0,
        'last_seq': None
    }
    start = time.time()
    last_print = start
    print(f"[+] started capture on {port} @ {baud} baud, duration={duration}s, out={outdir}")
    try:
        while True:
            if duration is not None and (time.time() - start) >= duration:
                print("[*] duration reached, stopping capture")
                break
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
                parse_chunk_buffer(buf, ch_data, stats)
            # periodic status
            now = time.time()
            if now - last_print > 2.0:
                last_print = now
                print(f"[i] parsed={stats['packets_parsed']} bad_crc={stats['bad_crc']} seq_lost={stats['seq_lost']} buf_len={len(buf)}")
    except KeyboardInterrupt:
        print("[*] interrupted by user")
    finally:
        ser.close()
        # Save time series
        save_time_series(outdir, ch_data)
        # Write summary
        write_summary(outdir, stats, ch_data)
        # Compute spectra
        # if arrays are dB (is_db_input True), compute welch using conversion to Pa
        if len(ch_data) == 0:
            print("[!] No data collected.")
            return
        compute_and_save_welch(ch_data, fs=fs_db, outdir=outdir, nperseg=nperseg, is_db_input=is_db_input, output_units=output_units)
        if compute_linear:
            compute_and_save_linear(ch_data, fs=fs_db, outdir=outdir, nperseg=nperseg, is_db_input=is_db_input, mode=linear_mode)
        print("[+] done.")

def main():
    p = argparse.ArgumentParser(prog="uart_capture_and_spectrum.py")
    p.add_argument('--port', required=True, help='Serial port (COMx or /dev/ttyUSBx)')
    p.add_argument('--baud', type=int, default=921600, help='Baud rate (default 921600)')
    p.add_argument('--duration', type=float, default=10.0, help='Capture duration in seconds (default 10s); use 0 or omit to run until Ctrl+C')
    p.add_argument('--out', default='out', help='Output directory')
    p.add_argument('--fs', type=float, default=16000.0, help='Sampling rate of incoming dB samples (Hz)')
    p.add_argument('--nperseg', type=int, default=1024, help='nperseg for Welch')
    p.add_argument('--is-db', action='store_true', help='Input arrays are dB SPL (default True behavior), if unset script treats them as linear (Pa)')
    p.add_argument('--no-linear', dest='compute_linear', action='store_false', help='Do not compute linear PSD/ASD')
    p.add_argument('--linear-mode', choices=['psd','asd'], default='psd', help='If computing linear, compute "psd" or "asd"')
    p.add_argument('--output-units', choices=['pa', 'db'], default='pa',
                  help='Output units: pa (Pascal) or db (dB SPL)')
    args = p.parse_args()

    # Normalize duration semantics: if <=0 then run until Ctrl+C
    duration = args.duration if args.duration > 0 else None
    os.makedirs(args.out, exist_ok=True)
    run_capture(args.port, args.baud, duration, args.out, 
                fs_db=args.fs, nperseg=args.nperseg, 
                is_db_input=False, 
                compute_linear=args.compute_linear, 
                linear_mode=args.linear_mode,
                output_units=args.output_units)

if __name__ == '__main__':
    main()
