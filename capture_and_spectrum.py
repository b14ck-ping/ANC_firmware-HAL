#!/usr/bin/env python3
"""
capture_and_spectrum.py

Read raw MCU packets from serial, save raw ADC codes and (optionally) convert to Pascals,
compute & save averaged spectra (Welch) and time-domain plots.

Packet format expected (from firmware):
  [0xAA, 0x55]       - preamble (2 bytes)
  [id: uint8]        - channel id (0 = ref, 1 = err)
  [cnt: uint32 LE]   - sample counter (4 bytes)
  [q: int16 LE]      - ADC code or quantized value (2 bytes)
Total packet length = 9 bytes

Usage:
  python capture_and_spectrum.py --port COM5 --baud 921600 --duration 10 --out outdir --fs 32000

Key options:
  --save-raw-adc    : save raw ADC arrays (chan0_adc.npy, chan1_adc.npy)
  --convert-pa      : convert ADC -> Pa using provided constants and save chanX_pa.npy
  --v-bias 1.5      : V_BIAS (default 1.5 V)
  --mic-gain-db 40  : MIC gain in dB (default 40 dB)
  --mic-sens-dbv -44: mic sensitivity in dBV/Pa (default -44 dBV)
  --nperseg 1024    : nperseg for Welch
"""

import argparse
import serial
import time
import os
import struct
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

PRE0 = 0xAA
PRE1 = 0x55
PKT_LEN = 9  # as in firmware (pre0,pre1,id,uint32 cnt, int16 q)
P0 = 20e-6

# Default ADC/hardware constants (tune if needed)
VREF = 3.3
ADC_MAX = 4095.0  # 12-bit
DEFAULT_V_BIAS = 1.5
DEFAULT_MIC_GAIN_DB = 40.0
DEFAULT_MIC_SENS_DBV = -44.0

def adc_to_pa(adc_code, v_ref=VREF, adc_max=ADC_MAX, v_bias=DEFAULT_V_BIAS,
              mic_gain_db=DEFAULT_MIC_GAIN_DB, mic_sens_dbv=DEFAULT_MIC_SENS_DBV):
    """
    Convert ADC code(s) to Pascals.
    Works for scalar or numpy array input (returns numpy array for array input).
    """
    import numpy as _np

    # ensure numpy array for vectorized ops
    a = _np.asarray(adc_code, dtype=_np.float64)

    # ADC -> voltage
    v_adc = (a / float(adc_max)) * float(v_ref)           # V

    # remove DC bias
    v_sig = v_adc - float(v_bias)

    # microphone chain constants
    mic_gain_linear = 10.0 ** (float(mic_gain_db) / 20.0)
    mic_sens_v_per_pa = 10.0 ** (float(mic_sens_dbv) / 20.0)
    denom = mic_gain_linear * mic_sens_v_per_pa
    # avoid division by zero
    if denom == 0.0:
        denom = 1e-12

    p = v_sig / denom
    return p


def parse_buffer(buf, ch_adc, stats):
    """
    Parse full packets from buf (bytearray). Append raw q (int16) to ch_adc[channel].
    Update stats dict: packets_parsed, malformed.
    Returns True if parsed any.
    """
    i = 0
    parsed = False
    while i + PKT_LEN <= len(buf):
        # find preamble
        if buf[i] != PRE0 or buf[i+1] != PRE1:
            i += 1
            continue
        pkt = buf[i:i+PKT_LEN]
        # unpack: pre0,pre1,id,u32 cnt (little), i16 q (little)
        try:
            _, _, ch_id, cnt, = struct.unpack_from('<BBBI', pkt, 0)  # will take 7 bytes; then q
            # NOTE: struct.unpack_from with '<BBBI' reads 1+1+1+4 = 7 bytes; q afterwards
            q = struct.unpack_from('<h', pkt, 7)[0]  # int16 little at offset 7
        except Exception:
            stats['malformed'] = stats.get('malformed', 0) + 1
            i += 1
            continue
        ch_adc.setdefault(int(ch_id), []).append(int(q))
        stats['packets_parsed'] = stats.get('packets_parsed', 0) + 1
        parsed = True
        i += PKT_LEN
    if i:
        del buf[:i]
    return parsed

def compute_and_save(ch_data_adc, outdir, fs, nperseg=1024, convert_to_pa=False,
                     v_bias=DEFAULT_V_BIAS, mic_gain_db=DEFAULT_MIC_GAIN_DB, mic_sens_dbv=DEFAULT_MIC_SENS_DBV,
                     output_units='pa'):
    os.makedirs(outdir, exist_ok=True)
    noverlap = nperseg // 2
    for ch, adc_list in ch_data_adc.items():
        adc_arr = np.asarray(adc_list, dtype=np.int32)
        np.save(os.path.join(outdir, f'chan{ch}_adc.npy'), adc_arr)
        print(f"[+] saved raw ADC chan{ch} ({len(adc_arr)} samples)")
        if convert_to_pa:
            pa = adc_to_pa(adc_arr.astype(float), v_ref=VREF, adc_max=ADC_MAX,
                           v_bias=v_bias, mic_gain_db=mic_gain_db, mic_sens_dbv=mic_sens_dbv)
            np.save(os.path.join(outdir, f'chan{ch}_pa.npy'), pa)
            x = pa
            ylab = 'Pa'
        else:
            x = adc_arr.astype(float)
            ylab = 'ADC counts'
        # Time plot
        t = np.arange(x.size) / fs
        plt.figure(figsize=(10,4))
        plt.plot(t, x, linewidth=0.6)
        plt.xlabel('Time (s)')
        plt.ylabel(ylab)
        plt.title(f'Channel {ch} time series ({len(x)} samples)')
        plt.grid(True)
        png_time = os.path.join(outdir, f'chan{ch}_time.png')
        plt.tight_layout()
        plt.savefig(png_time, dpi=150)
        plt.close()
        print(f"[+] saved time plot {png_time}")
        # Welch PSD
        # If we have Pa and want dB, convert: 10*log10(Pxx / P0**2)
        f, Pxx = signal.welch(x, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, detrend='constant', scaling='density')
        if convert_to_pa and output_units == 'db':
            Pxx_db = 10.0 * np.log10(np.maximum(Pxx, 1e-30) / (P0**2))
            data_to_plot = Pxx_db
            ylabel = 'PSD (dB re 20µPa²/Hz)'
            suffix = 'psd_db'
            np.save(os.path.join(outdir, f'chan{ch}_psd_freq.npy'), f)
            np.save(os.path.join(outdir, f'chan{ch}_psd_db.npy'), data_to_plot)
        else:
            data_to_plot = Pxx
            ylabel = 'PSD (unit²/Hz)'
            suffix = 'psd'
            np.save(os.path.join(outdir, f'chan{ch}_psd_freq.npy'), f)
            np.save(os.path.join(outdir, f'chan{ch}_psd.npy'), data_to_plot)
        plt.figure(figsize=(10,4))
        plt.semilogx(f, data_to_plot)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(ylabel)
        plt.title(f'Channel {ch} averaged spectrum')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.xlim(10, f.max())
        png_spec = os.path.join(outdir, f'chan{ch}_{suffix}.png')
        plt.tight_layout()
        plt.savefig(png_spec, dpi=150)
        plt.close()
        print(f"[+] saved PSD plot {png_spec}")

def main():
    p = argparse.ArgumentParser(prog="capture_and_spectrum.py")
    p.add_argument('--port', required=True, help='Serial port (COMx or /dev/ttyUSBx)')
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--duration', type=float, default=10.0, help='seconds (<=0 for until Ctrl+C)')
    p.add_argument('--out', default='out', help='output directory')
    p.add_argument('--fs', type=float, default=32000.0, help='sampling rate for time axis and PSD')
    p.add_argument('--nperseg', type=int, default=1024)
    p.add_argument('--save-raw-adc', action='store_true')
    p.add_argument('--convert-pa', action='store_true', help='convert ADC->Pa using constants')
    p.add_argument('--v-bias', type=float, default=DEFAULT_V_BIAS)
    p.add_argument('--mic-gain-db', type=float, default=DEFAULT_MIC_GAIN_DB)
    p.add_argument('--mic-sens-dbv', type=float, default=DEFAULT_MIC_SENS_DBV)
    p.add_argument('--output-units', choices=['pa','db','adc'], default='pa')
    args = p.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.1)
    buf = bytearray()
    ch_adc = {}
    stats = {'packets_parsed': 0, 'malformed': 0}
    start = time.time()
    last_print = start
    print(f"[+] capture start on {args.port} @{args.baud} baud, duration={args.duration}s -> {args.out}")
    try:
        while True:
            if args.duration > 0 and (time.time() - start) >= args.duration:
                print("[*] duration reached")
                break
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
                parse_buffer(buf, ch_adc, stats)
            now = time.time()
            if now - last_print > 2.0:
                last_print = now
                print(f"[i] parsed={stats['packets_parsed']} malformed={stats['malformed']} buf_len={len(buf)}")
    except KeyboardInterrupt:
        print("[*] interrupted by user")
    finally:
        ser.close()
        if len(ch_adc) == 0:
            print("[!] No data collected")
            return
        # Save raw ADC arrays if requested
        if args.save_raw_adc:
            os.makedirs(args.out, exist_ok=True)
            for ch, arr in ch_adc.items():
                np.save(os.path.join(args.out, f'chan{ch}_adc.npy'), np.asarray(arr, dtype=np.int32))
                print(f"[+] saved chan{ch}_adc.npy ({len(arr)} samples)")
        # Compute and save plots & PSD
        compute_and_save(ch_adc, args.out, fs=args.fs, nperseg=args.nperseg,
                         convert_to_pa=args.convert_pa, v_bias=args.v_bias,
                         mic_gain_db=args.mic_gain_db, mic_sens_dbv=args.mic_sens_dbv,
                         output_units=args.output_units)
        # Summary file
        with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
            f.write(f"packets_parsed={stats['packets_parsed']}\n")
            f.write(f"malformed={stats['malformed']}\n")
            for ch, arr in ch_adc.items():
                f.write(f"chan{ch}_samples={len(arr)}\n")
        print(f"[+] done. outputs in {args.out}")

if __name__ == '__main__':
    main()
