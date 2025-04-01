#include <chrono>
#include <iomanip>
#include <omp.h>
#include <openssl/sha.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

std::string sha1_hash(const std::string &input) {
  unsigned char hash[SHA_DIGEST_LENGTH];
  SHA1(reinterpret_cast<const unsigned char *>(input.c_str()), input.length(),
       hash);

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
  std::vector<std::tuple<std::string, int>> hashes;
  hashes.reserve(max_hashes);

  std::vector<std::vector<std::tuple<std::string, int>>> thread_results(
      omp_get_max_threads());

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    auto &thread_hashes = thread_results[thread_id];
    thread_hashes.reserve(max_hashes / omp_get_num_threads());

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

PYBIND11_MODULE(fingerprint_pybind, m) {
  m.doc() = "pybind11 extension for audio fingerprinting";

  m.def(
      "generate_hashes", &generate_hashes, py::arg("peaks"),
      py::arg("fan_value") = 15, py::arg("peak_sort") = true,
      py::arg("min_hash_time_delta") = 0, py::arg("max_hash_time_delta") = 200,
      py::arg("fingerprint_reduction") = 20,
      "Generate fingerprint hashes from peaks with corresponding time offsets");
}
