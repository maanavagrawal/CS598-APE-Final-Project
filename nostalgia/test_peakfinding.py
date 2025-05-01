import os
import sys
import time
import numpy as np
from typing import List, Tuple
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.filters import maximum_filter

# Add project root to path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the functions to compare
from nostalgia.fingerprint_pybind import get_2d_peaks_parallel as peakfinding_parallel

def load_audio_file(file_path: str):
    """Load an audio file and return the audio data."""
    print(f"Loading audio file: {file_path}")
    rate, data = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        print(f"Converting stereo to mono (channels: {data.shape[1]})")
        data = data.mean(axis=1)
    
    print(f"Audio loaded: {len(data)} samples, {rate} Hz")
    return data, rate

def create_spectrogram(audio_data: np.ndarray, sample_rate: int):
    """Create a spectrogram from audio data."""
    print("Creating spectrogram...")
    # STFT parameters
    nperseg = 2048
    noverlap = 1024
    
    # Compute spectrogram
    freqs, times, spec = signal.spectrogram(
        audio_data,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False
    )
    
    # Convert to log scale (dB)
    spec = np.log(np.abs(spec) + 1e-10)
    
    print(f"Spectrogram created: shape={spec.shape}, min={spec.min():.2f}, max={spec.max():.2f}")
    return spec

def get_2D_peaks_python(arr2D, amp_min=10):
    """Python implementation of peak finding using maximum filter."""
    # Apply maximum filter
    print("Running Python maximum filter implementation...")
    start_time = time.time()
    
    # Find all local maxima
    max_filtered = maximum_filter(arr2D, size=3, mode='constant', cval=0)
    peak_mask = (arr2D == max_filtered) & (arr2D > amp_min)
    
    # Get peak coordinates
    peak_coords = np.where(peak_mask)
    peaks = list(zip(peak_coords[0], peak_coords[1]))
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Python implementation found {len(peaks)} peaks in {execution_time:.4f} seconds")
    return peaks, execution_time

def compare_implementations(spec, fraction=0.1, condition=2, amp_min=10):
    """Compare Python and C++ implementations."""
    print("\n" + "="*80)
    print(f"COMPARISON WITH PARAMETERS: fraction={fraction}, condition={condition}, amp_min={amp_min}")
    print("="*80)
    
    # Run Python implementation
    py_peaks, py_time = get_2D_peaks_python(spec, amp_min=amp_min)
    
    # Run C++ implementation
    print("Running C++ implementation...")
    start_time = time.time()
    cpp_peaks = peakfinding_parallel(spec, amp_min=amp_min)
    cpp_time = time.time() - start_time
    
    print(f"C++ implementation found {len(cpp_peaks)} peaks in {cpp_time:.4f} seconds")
    
    # Convert to sets for comparison
    py_peaks_set = set(py_peaks)
    cpp_peaks_set = set(cpp_peaks)
    
    # Find common and unique peaks
    common_peaks = py_peaks_set.intersection(cpp_peaks_set)
    only_in_py = py_peaks_set - cpp_peaks_set
    only_in_cpp = cpp_peaks_set - py_peaks_set
    
    # Calculate metrics
    if len(py_peaks) > 0:
        recall = len(common_peaks) / len(py_peaks)
    else:
        recall = 1.0 if len(cpp_peaks) == 0 else 0.0
    
    if len(cpp_peaks) > 0:
        precision = len(common_peaks) / len(cpp_peaks)
    else:
        precision = 1.0 if len(py_peaks) == 0 else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    
    speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
    
    # Print detailed comparison
    print("\nRESULTS:")
    print(f"  Python peaks: {len(py_peaks)}")
    print(f"  C++ peaks: {len(cpp_peaks)}")
    print(f"  Common peaks: {len(common_peaks)}")
    print(f"  Only in Python: {len(only_in_py)}")
    print(f"  Only in C++: {len(only_in_cpp)}")
    print("\nMETRICS:")
    print(f"  Precision: {precision:.4f} (percentage of C++ peaks that match Python)")
    print(f"  Recall: {recall:.4f} (percentage of Python peaks found by C++)")
    print(f"  F1 Score: {f1_score:.4f}")
    print(f"  Speedup: {speedup:.2f}x (Python time / C++ time)")
    
    # Print sample of peaks for debugging
    if len(py_peaks) > 0:
        print("\nSAMPLE PYTHON PEAKS (first 5):")
        for i, peak in enumerate(list(py_peaks)[:5]):
            print(f"  {i+1}. {peak} (value: {spec[peak[0], peak[1]]:.2f})")
    
    if len(cpp_peaks) > 0:
        print("\nSAMPLE C++ PEAKS (first 5):")
        for i, peak in enumerate(list(cpp_peaks)[:5]):
            print(f"  {i+1}. {peak} (value: {spec[peak[0], peak[1]]:.2f})")
    
    if len(only_in_py) > 0:
        print("\nSAMPLE PEAKS ONLY IN PYTHON (first 5):")
        for i, peak in enumerate(list(only_in_py)[:5]):
            print(f"  {i+1}. {peak} (value: {spec[peak[0], peak[1]]:.2f})")
    
    if len(only_in_cpp) > 0:
        print("\nSAMPLE PEAKS ONLY IN C++ (first 5):")
        for i, peak in enumerate(list(only_in_cpp)[:5]):
            print(f"  {i+1}. {peak} (value: {spec[peak[0], peak[1]]:.2f})")
    
    return {
        'py_peaks': len(py_peaks),
        'cpp_peaks': len(cpp_peaks),
        'common': len(common_peaks),
        'only_py': len(only_in_py),
        'only_cpp': len(only_in_cpp),
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'py_time': py_time,
        'cpp_time': cpp_time,
        'speedup': speedup
    }

def run_test_suite(audio_file_path):
    """Run a suite of tests with different parameters."""
    # Load audio and create spectrogram
    audio_data, sample_rate = load_audio_file(audio_file_path)
    spec = create_spectrogram(audio_data, sample_rate)
    
    # Test with different parameters
    print("\nRunning test suite with different parameters...")
    
    # Default parameters
    compare_implementations(spec, fraction=0.1, condition=2, amp_min=10)
    
    # Different fractions
    compare_implementations(spec, fraction=0.05, condition=2, amp_min=10)
    compare_implementations(spec, fraction=0.2, condition=2, amp_min=10)
    
    # Different conditions
    compare_implementations(spec, fraction=0.1, condition=0, amp_min=10)
    compare_implementations(spec, fraction=0.1, condition=1, amp_min=10)
    
    # Different amp_min values
    compare_implementations(spec, fraction=0.1, condition=2, amp_min=5)
    compare_implementations(spec, fraction=0.1, condition=2, amp_min=20)
    
    print("\nTest suite completed.")

if __name__ == "__main__":
    # Default test file or use command line argument
    audio_file_path = "../test/sean_secs.wav"
    
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
    
    run_test_suite(audio_file_path)