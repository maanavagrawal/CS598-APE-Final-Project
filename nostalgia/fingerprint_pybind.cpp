#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <string>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace py = pybind11;

std::string sha1_hash(const std::string& input) {
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(input.c_str()), input.length(), hash);
    std::stringstream ss;
    for (int i = 0; i < SHA_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

std::tuple<std::vector<std::tuple<std::string, int>>, double> generate_hashes(
    const std::vector<std::tuple<int, int>>& peaks,
    int fan_value = 15,
    bool peak_sort = true,
    int min_hash_time_delta = 0,
    int max_hash_time_delta = 200,
    int fingerprint_reduction = 20
) {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    std::vector<std::tuple<int, int>> sorted_peaks;
    
    if (peak_sort) {
        sorted_peaks = peaks;
        std::sort(sorted_peaks.begin(), sorted_peaks.end(),
            [](const std::tuple<int, int>& a, const std::tuple<int, int>& b) {
                return std::get<1>(a) < std::get<1>(b);
            });
    } else {
        sorted_peaks = peaks;
    }

    std::vector<std::tuple<std::string, int>> hashes;
    
    for (size_t i = 0; i < sorted_peaks.size(); i++) {
        for (int j = 1; j < fan_value; j++) {
            if (i + j < sorted_peaks.size()) {
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
                    hashes.emplace_back(hash_str, t1);
                }
            }
        }
    }
    auto end_time = high_resolution_clock::now();
    double execution_time = duration_cast<microseconds>(end_time - start_time).count() / 1000000.0;
    
    return std::make_tuple(hashes, execution_time);
}

PYBIND11_MODULE(fingerprint_pybind, m) {
    m.doc() = "pybind11 extension for audio fingerprinting";
    
    m.def("generate_hashes", &generate_hashes, 
          py::arg("peaks"),
          py::arg("fan_value") = 15,
          py::arg("peak_sort") = true,
          py::arg("min_hash_time_delta") = 0,
          py::arg("max_hash_time_delta") = 200,
          py::arg("fingerprint_reduction") = 20,
          "Generate fingerprint hashes from peaks with corresponding time offsets");
}