#!/usr/bin/env python3
"""
play_and_save.py

Load .npy time series (either raw ADC codes or Pascals), optionally convert ADC->Pa,
safely convert to normalized audio (-1..1) and play/save to WAV.

Usage:
  python play_and_save.py --file out/chan0_adc.npy --fs 32000 --mode adc --v-bias 1.5 --save out/chan0_safe.wav --play

  python play_and_save.py --file out/chan0_pa.npy --fs 32000 --mode pa --save out/chan0_safe.wav --play

Key options:
  --mode {adc,pa}    : treat input as raw ADC codes or Pascals
  --v-bias           : V_BIAS for ADC->Pa conversion
  --mic-gain-db      : mic gain in dB
  --mic-sens-dbv     : mic sensitivity dBV/Pa
  --normalize {peak,rms,none} : normalization strategy (default peak)
  --softclip/--no-softclip   : apply conservative tanh limiter (default True)
  --play             : play via sounddevice
  --save             : path to save WAV via soundfile
  --peak-target      : peak target for normalization (0..1)
"""

import argparse
import numpy as np
import os
import soundfile as sf
import sounddevice as sd
from scipy import signal

P0 = 20e-6
VREF = 3.3
ADC_MAX = 4095.0

def adc_to_pa(adc_arr, v_bias=1.5, mic_gain_db=40.0, mic_sens_dbv=-44.0):
    v_adc = (adc_arr.astype(float) / ADC_MAX) * VREF
    v_sig = v_adc - float(v_bias)
    mic_gain_lin = 10.0 ** (float(mic_gain_db)/20.0)
    mic_sens_v_per_pa = 10.0 ** (float(mic_sens_dbv)/20.0)
    denom = mic_gain_lin * mic_sens_v_per_pa
    denom = denom if denom != 0.0 else 1e-12
    pa = v_sig / denom
    return pa

def pa_to_audio_safe(pa_arr, normalize='peak', peak_target=0.95, softclip=True, soft_drive=1.0):
    # remove DC
    x = pa_arr - np.mean(pa_arr)
    if normalize == 'peak':
        peak = np.max(np.abs(x)) + 1e-30
        scale = peak_target / peak
        audio = x * scale
    elif normalize == 'rms':
        rms = np.sqrt(np.mean(x**2)) + 1e-30
        target_rms = peak_target * 0.1  # fallback small RMS target
        scale = target_rms / rms
        audio = x * scale
    else:
        # no normalization, just scale to reasonable range (tiny)
        audio = x
    if softclip:
        # conservative tanh limiter
        audio = np.tanh(audio * soft_drive)
        # rescale so peak equals peak_target
        max_abs = np.max(np.abs(audio)) + 1e-30
        audio = audio / max_abs * peak_target
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def analyze_and_print(arr, fs, name='signal'):
    N = len(arr)
    dur = N / fs
    mean = np.mean(arr)
    rms = np.sqrt(np.mean((arr - 0.0)**2))
    peak = np.max(np.abs(arr))
    p2p = np.ptp(arr)
    print(f"[{name}] samples={N}, dur={dur:.3f}s, mean={mean:.6e}, rms={rms:.6e}, peak={peak:.6e}, p2p={p2p:.6e}")
    try:
        print(f"[{name}] est RMS dB SPL: {20.0*np.log10(max(rms,1e-20)/P0):.2f} dB")
    except Exception:
        pass

def main():
    p = argparse.ArgumentParser(prog="play_and_save.py")
    p.add_argument('--file', required=True, help=".npy file (adc or pa)")
    p.add_argument('--mode', choices=['adc','pa'], default='pa', help='input type')
    p.add_argument('--fs', type=float, required=True, help='sample rate for playback and analysis')
    p.add_argument('--v-bias', type=float, default=1.5)
    p.add_argument('--mic-gain-db', type=float, default=40.0)
    p.add_argument('--mic-sens-dbv', type=float, default=-44.0)
    p.add_argument('--normalize', choices=['peak','rms','none'], default='peak')
    p.add_argument('--softclip', dest='softclip', action='store_true')
    p.add_argument('--no-softclip', dest='softclip', action='store_false')
    p.set_defaults(softclip=True)
    p.add_argument('--peak-target', type=float, default=0.95)
    p.add_argument('--play', action='store_true')
    p.add_argument('--save', default=None, help='save WAV to path (e.g. out.wav)')
    args = p.parse_args()

    arr = np.load(args.file)
    print(f"Loaded {args.file}, dtype={arr.dtype}, shape={arr.shape}")
    if args.mode == 'adc':
        adc = arr.astype(np.int32)
        analyze_and_print(adc, args.fs, name='adc_raw')
        pa = adc_to_pa(adc, v_bias=args.v_bias, mic_gain_db=args.mic_gain_db, mic_sens_dbv=args.mic_sens_dbv)
        analyze_and_print(pa, args.fs, name='pa_converted')
    else:
        pa = arr.astype(np.float64)
        analyze_and_print(pa, args.fs, name='pa_input')

    audio = pa_to_audio_safe(pa, normalize=args.normalize, peak_target=args.peak_target, softclip=args.softclip, soft_drive=1.0)

    if args.save:
        # convert float32 -1..1 to PCM float WAV (soundfile handles float)
        sf.write(args.save, audio.astype(np.float32), int(args.fs))
        print(f"[+] saved WAV {args.save}")

    if args.play:
        try:
            print("[*] playing... (blocking)")
            sd.play(audio, int(args.fs))
            sd.wait()
            print("[*] playback finished")
        except KeyboardInterrupt:
            sd.stop()
            print("[*] playback stopped by user")

if __name__ == '__main__':
    main()
