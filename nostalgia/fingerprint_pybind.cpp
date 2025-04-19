#include "hash/sha1.h"
#include <chrono>
#include <iomanip>
#include <omp.h>
// #include <openssl/sha.h>
#include <execution>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/numpy.h>
#include <algorithm>

namespace py = pybind11;

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
    std::sort(sorted_peaks.begin(), sorted_peaks.end(),
              [](const std::tuple<int, int> &a, const std::tuple<int, int> &b) {
                return std::get<1>(a) < std::get<1>(b);
              });
  } else {
    sorted_peaks = peaks;
  }

  size_t total_peaks = sorted_peaks.size();
  size_t max_hashes = total_peaks * (fan_value - 1);

  std::vector<size_t> offsets(total_peaks, 0);
// compute offsets
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
    for (int j = 1; j < fan_value; j++) {
      size_t offset = result[i];
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
generate_hashes_test(const std::vector<std::tuple<int, int>> &peaks,
                     int fan_value = 15, bool peak_sort = true,
                     int min_hash_time_delta = 0, int max_hash_time_delta = 200,
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


std::tuple<py::array_t<bool>, py::array_t<double>> 
get_2D_peaks_cpp(py::array_t<double> spectrogram, double fraction = 0.1, int condition = 2, double amp_min = 10) {
    py::buffer_info buf = spectrogram.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be 2");
    }
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    double* data = static_cast<double*>(buf.ptr);
    
    py::array_t<bool> peak_locations = py::array_t<bool>({rows, cols});
    py::array_t<double> peak_values = py::array_t<double>({rows, cols});
    
    py::buffer_info peak_loc_buf = peak_locations.request();
    py::buffer_info peak_val_buf = peak_values.request();
    
    bool* peak_loc_ptr = static_cast<bool*>(peak_loc_buf.ptr);
    double* peak_val_ptr = static_cast<double*>(peak_val_buf.ptr);
  
    std::fill(peak_loc_ptr, peak_loc_ptr + rows * cols, false);
    std::fill(peak_val_ptr, peak_val_ptr + rows * cols, 0.0);
    if (condition == 0 || condition == 2) {
        int distance = static_cast<int>(fraction * rows);
        #pragma omp parallel for
        for (size_t col = 0; col < cols; col++) {
            std::vector<std::pair<double, size_t>> col_values(rows);

            for (size_t row = 0; row < rows; row++) {
                col_values[row] = {data[row * cols + col], row};
            }

            std::sort(col_values.begin(), col_values.end(), 
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<bool> used(rows, false);
            for (const auto& [val, idx] : col_values) {
                if (val <= amp_min) break;
                bool can_use = true;
                for (int d = -distance; d <= distance; d++) {
                    int check_idx = idx + d;
                    if (check_idx >= 0 && check_idx < static_cast<int>(rows) && used[check_idx]) {
                        can_use = false;
                        break;
                    }
                }
                if (can_use) {
                    used[idx] = true;
                    if (condition == 0) {
                        peak_loc_ptr[idx * cols + col] = true;
                        peak_val_ptr[idx * cols + col] = val;
                    }
                }
            }
        }
    }
    
    if (condition == 1 || condition == 2) {
        int distance = static_cast<int>(fraction * cols);

        #pragma omp parallel for
        for (size_t row = 0; row < rows; row++) {
            std::vector<std::pair<double, size_t>> row_values(cols);
            for (size_t col = 0; col < cols; col++) {
                row_values[col] = {data[row * cols + col], col};
            }
            std::sort(row_values.begin(), row_values.end(), 
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<bool> used(cols, false);
            for (const auto& [val, idx] : row_values) {
                if (val <= amp_min) break;
                bool can_use = true;
                for (int d = -distance; d <= distance; d++) {
                    int check_idx = idx + d;
                    if (check_idx >= 0 && check_idx < static_cast<int>(cols) && used[check_idx]) {
                        can_use = false;
                        break;
                    }
                }
                
                if (can_use) {
                    used[idx] = true;
                    if (condition == 1) {
                        peak_loc_ptr[row * cols + idx] = true;
                        peak_val_ptr[row * cols + idx] = val;
                    } else if (condition == 2 && peak_loc_ptr[row * cols + idx]) {
                        peak_val_ptr[row * cols + idx] = val;
                    }
                }
            }
        }
    }
    
    if (condition == 2) {
        #pragma omp parallel for collapse(2)
        for (size_t row = 0; row < rows; row++) {
            for (size_t col = 0; col < cols; col++) {
                size_t idx = row * cols + col;
                double val = data[idx];
                if (peak_loc_ptr[idx] && val > amp_min) {
                    peak_loc_ptr[idx] = true;
                    peak_val_ptr[idx] = val;
                } else {
                    peak_loc_ptr[idx] = false;
                    peak_val_ptr[idx] = 0.0;
                }
            }
        }
    }
    
    return std::make_tuple(peak_locations, peak_values);
}

std::vector<std::tuple<int, int>> 
peaks_to_coordinates(py::array_t<bool> peak_locations, py::array_t<double> spectrogram, double amp_min = 10) {
    py::buffer_info loc_buf = peak_locations.request();
    py::buffer_info spec_buf = spectrogram.request();
    
    if (loc_buf.ndim != 2 || spec_buf.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2-dimensional");
    }
    
    size_t rows = loc_buf.shape[0];
    size_t cols = loc_buf.shape[1];
    
    bool* loc_ptr = static_cast<bool*>(loc_buf.ptr);
    double* spec_ptr = static_cast<double*>(spec_buf.ptr);
    
    std::vector<std::tuple<int, int>> coordinates;
    
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            size_t idx = row * cols + col;
            if (loc_ptr[idx] && spec_ptr[idx] > amp_min) {
                coordinates.emplace_back(row, col);
            }
        }
    }
    
    return coordinates;
}

PYBIND11_MODULE(fingerprint_pybind, m) {
  m.doc() = "pybind11 extension for audio fingerprinting";

  m.def("generate_hashes", &generate_hashes, py::arg("peaks"),
        py::arg("fan_value") = 15, py::arg("peak_sort") = true,
        py::arg("min_hash_time_delta") = 0,
        py::arg("max_hash_time_delta") = 200,
        py::arg("fingerprint_reduction") = 20,
        "Generate fingerprint hashes from peaks with corresponding time "
        "offsets");
  
  // Add the new functions
  m.def("get_2D_peaks_cpp", &get_2D_peaks_cpp, 
        py::arg("spectrogram"), 
        py::arg("fraction") = 0.1, 
        py::arg("condition") = 2,
        py::arg("amp_min") = 10,
        "Find peaks in a spectrogram using a similar approach to scipy.signal.find_peaks");
  
  m.def("peaks_to_coordinates", &peaks_to_coordinates,
        py::arg("peak_locations"),
        py::arg("spectrogram"),
        py::arg("amp_min") = 10,
        "Convert peak locations to frequency-time coordinate pairs");
}
