import hashlib
from typing import List, Tuple
from operator import itemgetter

DEF DEFAULT_FAN_VALUE = 15
DEF MIN_HASH_TIME_DELTA = 0
DEF MAX_HASH_TIME_DELTA = 200
DEF FINGERPRINT_REDUCTION = 20
DEF PEAK_SORT = True

def generate_hashes_cython(peaks: List[Tuple[int, int]], fan_value: int = DEFAULT_FAN_VALUE) -> List[Tuple[str, int]]:
    cdef:
        int i, j, freq1, freq2, t1, t2, t_delta
        int n_peaks = len(peaks)
        list hashes = []
        bytes hashstr
        str hash_str
        int idx_freq = 0
        int idx_time = 1
        tuple peak_i, peak_j
    
    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))
    
    for i in range(n_peaks):
        peak_i = peaks[i]
        for j in range(1, fan_value):
            if (i + j) < n_peaks:
                peak_j = peaks[i + j]
                freq1 = peak_i[idx_freq]
                freq2 = peak_j[idx_freq]
                t1 = peak_i[idx_time]
                t2 = peak_j[idx_time]
                t_delta = t2 - t1
                
                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    hashstr = f"{freq1}|{freq2}|{t_delta}".encode('utf-8')
                    h = hashlib.sha1(hashstr)
                    hash_str = h.hexdigest()[:FINGERPRINT_REDUCTION]
                    hashes.append((hash_str, t1))
    
    return hashes 