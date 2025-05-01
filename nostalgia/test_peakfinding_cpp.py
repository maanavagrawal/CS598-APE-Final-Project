import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dejavu.logic.fingerprint import get_2D_peaks, get_2D_peaks_py
from nostalgia.fingerprint_pybind import (
    peak_finding_to_coordinates, 
    get_data_for_verification, 
    find_peaks_from_verified_data
)

def test_peakfinding_cpp():
    # Load the test data
    wav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp3", "azan_test.wav")
    print(f"Loading WAV file: {wav_path}")
    sample_rate, audio_data = wavfile.read(wav_path)
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Just use a smaller segment to make testing faster
    segment_length = 5 * sample_rate  # 5 seconds
    audio_segment = audio_data[:segment_length] if len(audio_data) > segment_length else audio_data
    
    print(f"Audio: sample rate={sample_rate}Hz, duration={len(audio_segment)/sample_rate:.2f}s")
    
    # Create spectrogram
    window_size = 2048
    overlap = window_size // 2
    
    # Create spectrogram using scipy.signal
    frequencies, times, test_data = signal.spectrogram(
        audio_segment, 
        fs=sample_rate, 
        window='hann',
        nperseg=window_size, 
        noverlap=overlap,
        detrend=False
    )


    # Get the data for verification
    verified_data = get_data_for_verification(test_data)

    # Convert verified_data from list to numpy array if needed


    # Run the C++ peak finding function multiple times
    num_runs = 10
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        cpp_peaks = find_peaks_from_verified_data(verified_data)
        cpp_time = time.time() - start_time
        total_time += cpp_time
    print(f"C++ peak finding time: {total_time / num_runs:.4f} seconds")

test_peakfinding_cpp()