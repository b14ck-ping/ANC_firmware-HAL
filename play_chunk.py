#!/usr/bin/env python3
"""
play_chunk.py

Play back a recorded chunk saved as .npy (samples in Pascals).
Converts pressure values to normalized audio (-1..1) for playback.

Usage examples:
  python play_chunk.py --file out/chan0.npy --fs 16000 --gain-db 20
  python play_chunk.py --file out/chan0.npy --fs 16000 --save-wav out/audio.wav
"""

import argparse
import os
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal

def plot_debug(data, fs, title="Debug Plot"):
    import matplotlib.pyplot as plt
    
    # Time domain
    plt.figure(figsize=(15,8))
    
    plt.subplot(211)
    t = np.arange(len(data)) / fs
    plt.plot(t, data)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Frequency domain
    plt.subplot(212)
    f, Pxx = signal.welch(data, fs, nperseg=1024)
    plt.semilogy(f, Pxx)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def load_array(path, channel=0):
    """Load array from file (supports .npy with Pa values)"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
        print(f"Loaded {arr.size} samples from {path}")
        print(f"Range: {arr.min():.2e} to {arr.max():.2e} Pa")
        return arr
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def pa_to_normalized_audio(pa_data, target_peak=0.7):
    """
    Convert ADC values to normalized audio (-1..1)
    1. Remove DC offset
    2. Scale to target amplitude
    3. Apply gentle limiting
    """
    # Remove DC offset
    data_centered = pa_data - np.mean(pa_data)
    
    # Calculate RMS and peak values
    rms = np.sqrt(np.mean(data_centered**2))
    peak = np.max(np.abs(data_centered))
    print(f"Original - RMS: {rms:.2e}, Peak: {peak:.2e}, Crest factor: {peak/rms:.1f}")
    
    # First scale to target RMS level (-20 dBFS)
    target_rms = 0.1  # -20 dBFS
    gain_rms = target_rms / rms
    normalized = data_centered * gain_rms
    
    # Apply soft knee limiting
    def soft_clip(x, threshold=0.7, knee=0.1):
        y = x.copy()
        # Soft knee region
        soft_region = (np.abs(x) > (threshold - knee)) & (np.abs(x) <= (threshold + knee))
        y[soft_region] = np.sign(x[soft_region]) * (
            threshold + knee * np.sin((np.pi/(4*knee)) * (x[soft_region] - threshold))
        )
        # Hard limit
        y[np.abs(x) > (threshold + knee)] = np.sign(x[np.abs(x) > (threshold + knee)]) * threshold
        return y
    
    # Apply soft limiting
    normalized = soft_clip(normalized, threshold=target_peak, knee=0.1)
    
    # Report final levels
    final_rms = np.sqrt(np.mean(normalized**2))
    final_peak = np.max(np.abs(normalized))
    print(f"Normalized - RMS: {final_rms:.3f}, Peak: {final_peak:.3f}, Crest factor: {final_peak/final_rms:.1f}")
    
    return normalized

def apply_gain(arr, gain_db):
    """Apply gain in dB to normalized audio"""
    if gain_db == 0:
        return arr
    factor = 10.0 ** (gain_db / 20.0)
    return arr * factor

def apply_filters(data, fs):
    """Apply audio cleanup filters"""
    # High-pass filter to remove DC and very low frequencies
    sos_hp = signal.butter(2, 20, 'hp', fs=fs, output='sos')
    data = signal.sosfilt(sos_hp, data)
    
    # Low-pass filter to remove high frequency noise
    sos_lp = signal.butter(2, 7000, 'lp', fs=fs, output='sos')
    data = signal.sosfilt(sos_lp, data)
    
    return data

def analyze_raw_adc(data, fs):
    """Analyze raw ADC data for common problems"""
    print("\nADC Analysis:")
    
    # Check DC offset
    mean_adc = np.mean(data)
    expected_dc = (V_BIAS / VREF) * ADC_MAX
    print(f"Mean ADC: {mean_adc:.1f} (expected ~{expected_dc:.1f})")
    
    # Check dynamic range usage
    p2p = np.ptp(data)
    print(f"Peak-to-peak: {p2p:.1f} ADC counts")
    
    # Check for stuck bits
    unique_values = np.unique(data)
    print(f"Unique ADC values: {len(unique_values)}")
    
    # Look for missing codes
    if len(unique_values) > 1:
        gaps = np.diff(np.sort(unique_values))
        max_gap = np.max(gaps)
        if max_gap > 1:
            print(f"Warning: Missing codes detected, max gap: {max_gap}")
    
    # Spectral analysis
    f, Pxx = signal.welch(data, fs, nperseg=1024)
    main_freq_idx = np.argmax(Pxx[1:]) + 1  # skip DC
    main_freq = f[main_freq_idx]
    print(f"Strongest frequency component: {main_freq:.1f} Hz")
    
    return mean_adc, p2p

def analyze_signal(data, fs):
    """Detailed signal analysis"""
    print("\nSignal Analysis:")
    
    # Basic statistics
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(data)/fs:.2f} seconds")
    
    # Amplitude distribution
    hist, bins = np.histogram(data, bins=50)
    max_bin = np.argmax(hist)
    mode_value = (bins[max_bin] + bins[max_bin + 1]) / 2
    print(f"Most common value (mode): {mode_value:.1f}")
    
    # Frequency analysis
    f, Pxx = signal.welch(data, fs, nperseg=2048)
    
    # Find main frequency components
    peak_indices = signal.find_peaks(Pxx)[0]
    peak_freqs = f[peak_indices]
    peak_powers = Pxx[peak_indices]
    
    # Sort by power
    sorted_idx = np.argsort(peak_powers)[-5:]  # top 5
    print("\nMain frequency components:")
    for idx in sorted_idx[::-1]:
        freq = peak_freqs[idx]
        power = peak_powers[idx]
        print(f"  {freq:.1f} Hz (power: {power:.2e})")
    
    return f, Pxx

def main():
    p = argparse.ArgumentParser(description="Play recorded audio data")
    p.add_argument('--file', '-f', required=True, help='Input file (.npy with Pa values)')
    p.add_argument('--fs', type=float, required=True, help='Sample rate (Hz)')
    p.add_argument('--gain-db', type=float, default=0.0, 
                  help='Additional gain in dB for playback')
    p.add_argument('--save-wav', metavar='OUT.WAV', default=None,
                  help='Save as WAV file')
    p.add_argument('--blocksize', type=int, default=1024,
                  help='Audio buffer size (default 1024)')
    p.add_argument('--no-filters', action='store_true',
                  help='Disable audio cleanup filters')
    args = p.parse_args()

    # Load raw data
    raw_data = load_array(args.file)
    
    # Analyze raw signal
    print("\n=== Raw Signal Analysis ===")
    f_raw, Pxx_raw = analyze_signal(raw_data, args.fs)
    
    # Center the signal (remove DC)
    data_centered = raw_data - np.mean(raw_data)
    
    # Apply filters unless disabled
    if not args.no_filters:
        print("\nApplying audio cleanup filters...")
        # High-pass filter (remove below 100 Hz)
        sos_hp = signal.butter(4, 100, 'hp', fs=args.fs, output='sos')
        data_filtered = signal.sosfilt(sos_hp, data_centered)
        
        # Band-stop filter for possible 50/60Hz noise
        for freq in [50, 60]:
            sos_bs = signal.butter(4, [freq-2, freq+2], 'bandstop', fs=args.fs, output='sos')
            data_filtered = signal.sosfilt(sos_bs, data_filtered)
        
        # Low-pass filter (anti-aliasing)
        sos_lp = signal.butter(4, args.fs/2.5, 'lp', fs=args.fs, output='sos')
        data_filtered = signal.sosfilt(sos_lp, data_filtered)
    else:
        data_filtered = data_centered
    
    # Analyze filtered signal
    print("\n=== Filtered Signal Analysis ===")
    f_filt, Pxx_filt = analyze_signal(data_filtered, args.fs)
    
    # Convert to audio
    audio = pa_to_normalized_audio(data_filtered)
    
    # Plot comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.plot(np.arange(1000)/args.fs, raw_data[:1000], label='Raw')
    plt.plot(np.arange(1000)/args.fs, data_filtered[:1000], label='Filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(212)
    plt.semilogy(f_raw, Pxx_raw, label='Raw')
    plt.semilogy(f_filt, Pxx_filt, label='Filtered')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Save and play
    if args.save_wav:
        sf.write(args.save_wav, audio, int(args.fs))
        print(f"\nSaved WAV file: {args.save_wav}")
    
    try:
        print("\nPlaying... (Ctrl+C to stop)")
        sd.play(audio, args.fs, blocking=True)
    except KeyboardInterrupt:
        sd.stop()
        print("\nPlayback stopped.")
        
    # Plot debug before filtering
    plot_debug(raw_data, args.fs, "Raw Signal")  # было pa_data
    # Plot after filtering
    plot_debug(audio, args.fs, "Processed Signal")

# ADC characteristics (to be defined based on actual hardware)
VREF = 3.3       # опорное напряжение АЦП
ADC_MAX = 4095.0 # максимальный код 12-бит АЦП
V_BIAS = 1.5     # DC смещение микрофона (измеренное)

if __name__ == '__main__':
    main()