#include "hash/sha1.h"
#include <chrono>
#include <iomanip>
#include <omp.h>
// #include <openssl/sha.h>
#include <algorithm>
#include <execution>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <limits>
#include <iostream>

namespace py = pybind11;

// Forward declarations
std::vector<std::vector<double>> get_data_for_verification(py::array_t<double> spectrogram);
std::vector<std::tuple<int, int>> find_peaks_from_verified_data(const std::vector<std::vector<double>>& data, double amp_min);

std::string sha1_hash(const std::string &input) {
  unsigned char hash[SHA_DIGEST_LENGTH];
  //   SHA1(reinterpret_cast<const unsigned char *>(input.c_str()),
  //   input.length(),
  //        hash);
  SHA1((char *)hash, input.c_str(), input.length());

  std::string result(SHA_DIGEST_LENGTH * 2, '0');

  static constexpr char hex_chars[] = "0123456789abcdef";

  for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
    result[i * 2] = hex_chars[(hash[i] >> 4) & 0xF];
    result[i * 2 + 1] = hex_chars[hash[i] & 0xF];
  }

  return result;
}

std::tuple<std::vector<std::tuple<std::string, int>>, double>
generate_hashes(const std::vector<std::tuple<int, int>> &peaks,
                int fan_value = 15, bool peak_sort = true,
                int min_hash_time_delta = 0, int max_hash_time_delta = 200,
                int fingerprint_reduction = 20) {
  using namespace std::chrono;
  auto start_time = high_resolution_clock::now();
  std::vector<std::tuple<int, int>> sorted_peaks;

  if (peak_sort) {
    sorted_peaks = peaks;
    std::stable_sort(sorted_peaks.begin(), sorted_peaks.end(),
              [](const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
                return std::get<1>(a) < std::get<1>(b);
              });
  } else {
    sorted_peaks = peaks;
  }

  size_t total_peaks = sorted_peaks.size();
  std::vector<size_t> offsets(total_peaks, 0);

#pragma omp parallel for
  for (size_t i = 0; i < total_peaks; i++) {
    size_t num_hashes_per_peak = 0;
    for (int j = 1; j < fan_value; j++) {
      if (i + j < total_peaks) {
        int freq1 = std::get<0>(sorted_peaks[i]);
        int freq2 = std::get<0>(sorted_peaks[i + j]);
        int t1 = std::get<1>(sorted_peaks[i]);
        int t2 = std::get<1>(sorted_peaks[i + j]);
        int t_delta = t2 - t1;

        if (min_hash_time_delta <= t_delta && t_delta <= max_hash_time_delta) {
          num_hashes_per_peak++;
        }
      }
    }
    if (i < total_peaks - 1) {
      offsets[i + 1] = num_hashes_per_peak;
    }
  }

  std::vector<int> result(offsets.size());
  std::inclusive_scan(std::execution::par_unseq, offsets.begin(), offsets.end(),
                      result.begin(), std::plus<int>(), 0);

  // for (const auto &i : result) {
  //   std::cerr << i << std::endl;
  // }
  std::vector<std::tuple<std::string, int>> hashes(result.back());

#pragma omp parallel for schedule(dynamic, 100)
  for (size_t i = 0; i < total_peaks; i++) {
    size_t offset = result[i];
    for (int j = 1; j < fan_value; j++) {
      if (i + j < total_peaks) {
        int freq1 = std::get<0>(sorted_peaks[i]);
        int freq2 = std::get<0>(sorted_peaks[i + j]);
        int t1 = std::get<1>(sorted_peaks[i]);
        int t2 = std::get<1>(sorted_peaks[i + j]);
        int t_delta = t2 - t1;

        if (min_hash_time_delta <= t_delta && t_delta <= max_hash_time_delta) {
          std::stringstream key;
          key << freq1 << "|" << freq2 << "|" << t_delta;
          std::string hash_str = sha1_hash(key.str());
          hash_str = hash_str.substr(0, fingerprint_reduction);
          hashes[offset] = std::make_tuple(hash_str, t1);
          offset++;
        }
      }
    }
  }

  auto end_time = high_resolution_clock::now();
  double execution_time =
      duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;

  return std::make_tuple(hashes, execution_time);
}

std::tuple<std::vector<std::tuple<std::string, int>>, double>
generate_hashes_test_DO_NOT_USE(const std::vector<std::tuple<int, int>> &peaks,
                                int fan_value = 15, bool peak_sort = true,
                                int min_hash_time_delta = 0,
                                int max_hash_time_delta = 200,
                                int fingerprint_reduction = 20) {
  using namespace std::chrono;
  auto start_time = high_resolution_clock::now();
  std::vector<std::tuple<int, int>> sorted_peaks;

  if (peak_sort) {
    sorted_peaks = peaks;
    std::sort(sorted_peaks.begin(), sorted_peaks.end(),
              [](const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
                return std::get<1>(a) < std::get<1>(b);
              });
  } else {
    sorted_peaks = peaks;
  }

  size_t total_peaks = sorted_peaks.size();
  size_t max_hashes = total_peaks * (fan_value - 1);
  std::vector<std::tuple<std::string, int>> hashes;
  hashes.reserve(max_hashes);

  std::vector<std::vector<std::tuple<std::string, int>>> thread_results(
      omp_get_max_threads());

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    auto &thread_hashes = thread_results[thread_id];
    thread_hashes.reserve(max_hashes);

#pragma omp for schedule(dynamic, 100) nowait
    for (size_t i = 0; i < total_peaks; i++) {
      for (int j = 1; j < fan_value; j++) {
        if (i + j < total_peaks) {
          int freq1 = std::get<0>(sorted_peaks[i]);
          int freq2 = std::get<0>(sorted_peaks[i + j]);
          int t1 = std::get<1>(sorted_peaks[i]);
          int t2 = std::get<1>(sorted_peaks[i + j]);
          int t_delta = t2 - t1;

          if (min_hash_time_delta <= t_delta &&
              t_delta <= max_hash_time_delta) {
            std::stringstream key;
            key << freq1 << "|" << freq2 << "|" << t_delta;
            std::string hash_str = sha1_hash(key.str());
            hash_str = hash_str.substr(0, fingerprint_reduction);
            thread_hashes.emplace_back(hash_str, t1);
          }
        }
      }
    }
  }

  for (const auto &thread_result : thread_results) {
    hashes.insert(hashes.end(), thread_result.begin(), thread_result.end());
  }

  auto end_time = high_resolution_clock::now();
  double execution_time =
      duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;

  return std::make_tuple(hashes, execution_time);
}

std::tuple<std::vector<std::tuple<int, int>>, double>
peak_finding_to_coordinates(py::array_t<double> spectrogram,
                            double amp_min = 10) {
  using namespace std::chrono;
  auto start_time = high_resolution_clock::now();
  
  py::buffer_info buf = spectrogram.request();
  
  if (buf.ndim != 2) {
    throw std::runtime_error("Input must be a 2D array");
  }
  
  size_t rows = buf.shape[0];
  size_t cols = buf.shape[1];
  double *data = static_cast<double *>(buf.ptr);
  bool is_fortran_order = (buf.strides[0] < buf.strides[1]);
  
  std::vector<std::tuple<int, int>> peaks;
  peaks.reserve(rows * cols / 100); // Reserve space for expected peaks
  
  const int neighborhood_size = 5;
  const int total_neighbors = (2 * neighborhood_size + 1) * (2 * neighborhood_size + 1) - 1;
  
  // Pre-compute neighbor offsets for better memory access
  std::vector<std::pair<int, int>> neighbor_offsets;
  neighbor_offsets.reserve(total_neighbors);
  for (int di = -neighborhood_size; di <= neighborhood_size; di++) {
    for (int dj = -neighborhood_size; dj <= neighborhood_size; dj++) {
      if (di == 0 && dj == 0) continue;
      neighbor_offsets.emplace_back(di, dj);
    }
  }
  
  for (size_t i = neighborhood_size; i < rows - neighborhood_size; i++) {
    for (size_t j = neighborhood_size; j < cols - neighborhood_size; j++) {
      size_t idx = is_fortran_order ? (i + j * rows) : (i * cols + j);
      double center = data[idx];
      
      if (center <= amp_min) {
        continue;
      }
      
      bool is_max = true;
      for (const auto& [di, dj] : neighbor_offsets) {
        int ni = i + di;
        int nj = j + dj;
        
        if (0 <= ni && ni < static_cast<int>(rows) && 
            0 <= nj && nj < static_cast<int>(cols)) {
          size_t neighbor_idx = is_fortran_order ? (ni + nj * rows) : (ni * cols + nj);
          if (data[neighbor_idx] >= center) {
            is_max = false;
            break;
          }
        }
      }
      
      if (is_max) {
        peaks.emplace_back(i, j);
      }
    }
  }
  
  auto end_time = high_resolution_clock::now();
  double execution_time = duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;
  
  return std::make_tuple(peaks, execution_time);
}

// /* 
//  * DEPRECATED: Old implementation with memory layout issues 
//  * This function is kept for reference but not used
//  */
// std::vector<int> find_peaks(const double *spectrogram, double amp_min,
//                             size_t rows, size_t cols, size_t curr_row,
//                             size_t curr_col, bool row_wise = true) {
//   size_t max_peaks = (row_wise ? rows : cols) / 2;
//   std::vector<int> midpoints;
//   midpoints.reserve(max_peaks);

//   // Define neighborhood size (similar to PEAK_NEIGHBORHOOD_SIZE in Python)
//   const int neighborhood_size = 2;
  
//   size_t i = neighborhood_size; // Start after neighborhood
//   size_t i_max = row_wise ? rows - neighborhood_size : cols - neighborhood_size;

//   if (row_wise) {
//     while (i < i_max) {
//       size_t idx = curr_row * cols + i;
//       double curr_val = spectrogram[idx];
      
//       // Only consider values above amplitude threshold
//       if (curr_val > amp_min) {
//         bool is_peak = true;
        
//         // Check neighborhood in both directions
//         for (int offset = -neighborhood_size; offset <= neighborhood_size; offset++) {
//           if (offset == 0) continue; // Skip current point
          
//           size_t neighbor_idx = idx + offset;
//           if (spectrogram[neighbor_idx] >= curr_val) {
//             is_peak = false;
//             break;
//           }
//         }
        
//         if (is_peak) {
//           midpoints.push_back(i);
//         }
//       }
//       i++;
//     }
//   } else {
//     // column wise
//     while (i < i_max) {
//       size_t idx = i * cols + curr_col;
//       double curr_val = spectrogram[idx];
      
//       // Only consider values above amplitude threshold
//       if (curr_val > amp_min) {
//         bool is_peak = true;
        
//         // Check neighborhood in both directions
//         for (int offset = -neighborhood_size; offset <= neighborhood_size; offset++) {
//           if (offset == 0) continue; // Skip current point
          
//           size_t neighbor_idx = idx + offset * cols;
//           if (spectrogram[neighbor_idx] >= curr_val) {
//             is_peak = false;
//             break;
//           }
//         }
        
//         if (is_peak) {
//           midpoints.push_back(i);
//         }
//       }
//       i++;
//     }
//   }

//   return midpoints;
// }

PYBIND11_MODULE(fingerprint_pybind, m) {
  m.doc() = "pybind11 extension for audio fingerprinting";

  m.def("generate_hashes", &generate_hashes, py::arg("peaks"),
        py::arg("fan_value") = 15, py::arg("peak_sort") = true,
        py::arg("min_hash_time_delta") = 0,
        py::arg("max_hash_time_delta") = 200,
        py::arg("fingerprint_reduction") = 20,
        "Generate fingerprint hashes from peaks with corresponding time "
        "offsets");

  m.def("peak_finding_to_coordinates", &peak_finding_to_coordinates,
        py::arg("spectrogram"), py::arg("amp_min") = 10,
        "Find peaks in a spectrogram and return peaks with execution time");
}