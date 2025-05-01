import sys
import os
import time
import random
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dejavu.logic.fingerprint import generate_hashes_py
from nostalgia.fingerprint_pybind import generate_hashes as cpp_generate_hashes

PEAK_SORT = False
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
FINGERPRINT_REDUCTION = 20
FAN_VALUE = 15

def test_with_peaks(peaks, description=""):
    print(f"\nTesting with {len(peaks)} peaks{description}:")
    
    py_start = time.time()
    py_hashes = generate_hashes_py(peaks, fan_value=FAN_VALUE)
    py_time = time.time() - py_start

    cpp_result = cpp_generate_hashes(
        peaks, 
        fan_value=FAN_VALUE,
        peak_sort=PEAK_SORT,
        min_hash_time_delta=MIN_HASH_TIME_DELTA,
        max_hash_time_delta=MAX_HASH_TIME_DELTA,
        fingerprint_reduction=FINGERPRINT_REDUCTION
    )
    cpp_hashes, cpp_internal_time = cpp_result
    
    print(f"Python: {len(py_hashes)} hashes in {py_time:.4f}s")
    print(f"C++: {len(cpp_hashes)} hashes in {cpp_internal_time:.4f}s")
    print(f"Speedup: {py_time / cpp_internal_time:.2f}x")
    
    for i in range(len(py_hashes)):
        if i > len(cpp_hashes) or py_hashes[i] != cpp_hashes[i]:
            print(f"Mismatch at index {i}: Python={py_hashes[i]}, C++={cpp_hashes[i]}")
    
    return py_time, cpp_internal_time

if __name__ == "__main__":
    small_peaks = [(100, 10), (200, 30), (300, 50), (400, 70)]
    test_with_peaks(small_peaks, " (small test case)")
    
    random.seed(42)
    medium_peaks = [(random.randint(0, 1000), random.randint(0, 2000)) for _ in range(5000)]
    test_with_peaks(medium_peaks, " (medium test case)")
    
    random.seed(42)
    large_peaks = [(random.randint(0, 1000), random.randint(0, 2000)) for _ in range(100000)]
    test_with_peaks(large_peaks, " (large test case)")