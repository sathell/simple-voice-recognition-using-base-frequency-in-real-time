import numpy as np
import sounddevice as sd
from scipy.signal import get_window
from scipy.fft import fft
import sys
import time

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
THRESHOLD_F0 = 172.5

def HPS(fs, signal):
    window = get_window('hamming', len(signal))
    frame = signal * window

    n_fft = len(signal)
    spectrum = np.abs(fft(frame, n=n_fft))[:n_fft // 2]
    frequencies = np.linspace(0, fs / 2, len(spectrum))

    num_harmonics = 5
    hps_spectrum = np.copy(spectrum)

    for harmonic in range(2, num_harmonics + 1):
        downsampled = spectrum[::harmonic]
        hps_spectrum[:len(downsampled)] *= downsampled

    peak_index = np.argmax(hps_spectrum)
    f0_hps = frequencies[peak_index]
    while f0_hps < 85:
        hps_spectrum = np.delete(hps_spectrum, np.where(hps_spectrum == hps_spectrum[peak_index])[0][0])
        frequencies = np.delete(frequencies, np.where(frequencies == frequencies[peak_index])[0][0])
        peak_index = np.argmax(hps_spectrum)
        f0_hps = frequencies[peak_index]
    return f0_hps

def classify_gender(f0):
    if 50 < f0 < THRESHOLD_F0:
        return "M"
    elif THRESHOLD_F0 < f0 < 280:
        return "K"

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file= sys.stderr)
    signal = indata[:, 0]
    if len(signal) > 0:
        f0 = HPS(SAMPLE_RATE, signal)
        gender = classify_gender(f0)
        print(gender)

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=audio_callback):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopped.")
