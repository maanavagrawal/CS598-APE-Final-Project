import numpy as np
import sys
import os
import time
from pydub import AudioSegment

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dejavu.logic.fingerprint import get_2D_peaks, get_2D_peaks_py
from nostalgia.fingerprint_pybind import peak_finding_to_coordinates

def convert_to_python_types(peaks):
    """Convert numpy types to regular Python types for comparison"""
    return {(int(row), int(col)) for row, col in peaks}

def test_real_audio_spectrogram():
    """Test with a real audio file spectrogram"""
    try:
        import matplotlib.pyplot as plt
        from scipy import signal
        
        print("\nReal audio spectrogram test:")
        
        # Load the MP3 file
        mp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp3", "Brad-Sucks--Total-Breakdown.mp3")
        print(f"Loading MP3 file: {mp3_path}")
        
        audio = AudioSegment.from_mp3(mp3_path)
        sample_rate = audio.frame_rate
        audio_data = np.array(audio.get_array_of_samples())
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        segment_length = 5 * sample_rate
        audio_segment = audio_data[:segment_length] if len(audio_data) > segment_length else audio_data
        
        print(f"Audio: sample rate={sample_rate}Hz, duration={len(audio_segment)/sample_rate:.2f}s")
        
        window_size = 2048
        overlap = window_size // 2
        
        frequencies, times, spectrogram = signal.spectrogram(
            audio_segment, 
            fs=sample_rate, 
            window='hann',
            nperseg=window_size, 
            noverlap=overlap,
            detrend=False
        )
        
        spectrogram_db = 10 * np.log10(spectrogram + 1e-10)
        
        print(f"Spectrogram shape: {spectrogram_db.shape}")
        print(f"Value range: {np.min(spectrogram_db):.2f} to {np.max(spectrogram_db):.2f}")
        
        amp_min = np.mean(spectrogram_db) + 2 * np.std(spectrogram_db)
        print(f"Peak amplitude threshold: {amp_min:.2f} dB")
        
        spectrogram_db = spectrogram_db.astype(np.float64)
        
        py_start_time = time.time()
        py_peaks = get_2D_peaks_py(spectrogram_db, amp_min=amp_min)
        py_time = time.time() - py_start_time
        
        cpp_peaks, cpp_time = peak_finding_to_coordinates(spectrogram_db, amp_min=amp_min)
        
        py_peaks_set = convert_to_python_types(py_peaks)
        cpp_peaks_set = convert_to_python_types(cpp_peaks)
        
        overlap_peaks = py_peaks_set.intersection(cpp_peaks_set)
        
        print("\n----- PERFORMANCE RESULTS -----")
        print(f"Python implementation: {len(py_peaks_set)} peaks in {py_time:.6f} seconds")
        print(f"C++ implementation: {len(cpp_peaks_set)} peaks in {cpp_time:.6f} seconds")
        print(f"Speedup: {py_time/cpp_time:.2f}x faster")
        print(f"\n----- ACCURACY RESULTS -----")
        print(f"Overlap: {len(overlap_peaks)} peaks in common ({len(overlap_peaks)/len(py_peaks_set)*100:.1f}% of Python peaks)")
        
        print("\nSample Python peaks:")
        for i, peak in enumerate(list(py_peaks_set)[:5]):
            r, c = peak
            val = spectrogram_db[r, c]
            print(f"  Peak {i+1}: ({r}, {c}) - value: {val:.2f}")
            
        print("\nSample C++ peaks:")
        for i, peak in enumerate(list(cpp_peaks_set)[:5]):
            r, c = peak
            val = spectrogram_db[r, c]
            print(f"  Peak {i+1}: ({r}, {c}) - value: {val:.2f}")
        
    except ImportError as e:
        print(f"Skipping real audio test due to missing dependencies: {e}")
    except FileNotFoundError:
        print("Skipping real audio test: MP3 file not found")
    except Exception as e:
        print(f"Error in real audio test: {e}")

if __name__ == "__main__":
    print("Running peak finding comparison test...")
    test_real_audio_spectrogram()
    print("\nTest completed!") 