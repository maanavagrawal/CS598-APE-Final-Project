import hashlib
from operator import itemgetter
from typing import List, Tuple
import os
import multiprocessing

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter
    from cupyx.scipy.ndimage import binary_erosion, generate_binary_structure, iterate_structure
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cp.cuda.Device(0).use()
    USE_GPU = True
    print("Using GPU implementation")
except ImportError:
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import (binary_erosion,
                                        generate_binary_structure,
                                        iterate_structure)
    USE_GPU = False

from dejavu.config.settings import (CONNECTIVITY_MASK, DEFAULT_AMP_MIN,
                                    DEFAULT_FAN_VALUE, DEFAULT_FS,
                                    DEFAULT_OVERLAP_RATIO, DEFAULT_WINDOW_SIZE,
                                    FINGERPRINT_REDUCTION, MAX_HASH_TIME_DELTA,
                                    MIN_HASH_TIME_DELTA,
                                    PEAK_NEIGHBORHOOD_SIZE, PEAK_SORT)

try:
    from nostalgia.fingerprint_pybind import (
        generate_hashes as cpp_generate_hashes,
    )
    USE_CPP_IMPLEMENTATION = True
    print("Using C++ implementation")
except ImportError:
    USE_CPP_IMPLEMENTATION = False
    print("Using Python implementation")

try:
    from .fingerprint_cython import generate_hashes_cython
    USE_CYTHON = True
    print("Using Cython implementation")
except ImportError:
    USE_CYTHON = False
    print("Using Python implementation")


_gpu_initialized = False
_cp = None
_maximum_filter = None
_binary_erosion = None
_generate_binary_structure = None
_iterate_structure = None

def init_gpu():
    global _gpu_initialized, _cp, _maximum_filter, _binary_erosion, _generate_binary_structure, _iterate_structure
    if not _gpu_initialized:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import maximum_filter
            from cupyx.scipy.ndimage import binary_erosion, generate_binary_structure, iterate_structure
            
            process_id = multiprocessing.current_process().pid
            cp.cuda.Device(0).use()
            
            _cp = cp
            _maximum_filter = maximum_filter
            _binary_erosion = binary_erosion
            _generate_binary_structure = generate_binary_structure
            _iterate_structure = iterate_structure
            _gpu_initialized = True
            print(f"GPU initialized for process {process_id}")
            return True
        except Exception as e:
            print(f"GPU initialization failed for process {multiprocessing.current_process().pid}: {e}")
            _gpu_initialized = False
            return False
    return _gpu_initialized

def fingerprint(channel_samples: List[int],
                Fs: int = DEFAULT_FS,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.

    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """

    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))
    local_maxima = get_2D_peaks(arr2D, plot=False, amp_min=amp_min)

    if USE_CPP_IMPLEMENTATION:
        hashes, execution_time = cpp_generate_hashes(
            local_maxima, 
            fan_value=fan_value,
            peak_sort=PEAK_SORT,
            min_hash_time_delta=MIN_HASH_TIME_DELTA,
            max_hash_time_delta=MAX_HASH_TIME_DELTA,
            fingerprint_reduction=FINGERPRINT_REDUCTION
        )
        return hashes
    else:
        return generate_hashes_py(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN)\
        -> List[Tuple[List[int], List[int]]]:
    """
    Extract maximum peaks from the spectogram matrix (arr2D).

    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    USE_GPU = init_gpu()

    struct = _generate_binary_structure(2, CONNECTIVITY_MASK)
    if USE_GPU:
        neighborhood = _cp.ones((PEAK_NEIGHBORHOOD_SIZE, PEAK_NEIGHBORHOOD_SIZE), dtype=_cp.bool_)
    else:
        neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    if USE_GPU:
        try:
            arr2D_gpu = _cp.asarray(arr2D)
            neighborhood_gpu = _cp.asarray(neighborhood)
            
            local_max = _maximum_filter(arr2D_gpu, footprint=neighborhood_gpu) == arr2D_gpu
            
            background = (arr2D_gpu == 0)
            eroded_background = _binary_erosion(background, structure=neighborhood_gpu, iterations=1, brute_force=True)
            
            local_max = _cp.asnumpy(local_max)
            eroded_background = _cp.asnumpy(eroded_background)
        except Exception as e:
            print(f"GPU operation failed, falling back to CPU: {e}")
            local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
            background = (arr2D == 0)
            eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    else:
        local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
        background = (arr2D == 0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    detected_peaks = local_max != eroded_background

    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    amps = amps.flatten()
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    if plot:
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return list(zip(freqs_filter, times_filter))


def generate_hashes_py(peaks: List[Tuple[int, int]], fan_value: int = DEFAULT_FAN_VALUE) -> List[Tuple[str, int]]:
    """
    Hash list structure:
       sha1_hash[0:FINGERPRINT_REDUCTION]    time_offset
        [(e05b341a9b77a51fd26, 32), ... ]
    """
    if USE_CYTHON:
        return generate_hashes_cython(peaks, fan_value)
        
    idx_freq = 0
    idx_time = 1

    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))
        
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    hashstr = f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8')
                    h = hashlib.sha1(hashstr)
                    hash_str = h.hexdigest()[0:FINGERPRINT_REDUCTION]
                    hashes.append((hash_str, t1))

    return hashes