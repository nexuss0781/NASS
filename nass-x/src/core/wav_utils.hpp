#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace nass_x {

struct WavHeader {
    char riff[5] = "RIFF";
    uint32_t file_size = 0;
    char wave[5] = "WAVE";
    char fmt[5] = "fmt ";
    uint32_t fmt_size = 16;
    uint16_t audio_format = 3; // IEEE Float
    uint16_t num_channels = 1;
    uint32_t sample_rate = 44100;
    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 32; // Float32
    char data[5] = "data";
    uint32_t data_size = 0;
};

class WavUtils {
public:
    static bool write(const std::string& path, const std::vector<float>& samples, uint32_t sample_rate) {
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;

        WavHeader header;
        header.num_channels = 1;
        header.sample_rate = sample_rate;
        header.bits_per_sample = 32; // IEEE Float
        header.audio_format = 3; // IEEE Float
        header.block_align = header.num_channels * (header.bits_per_sample / 8);
        header.byte_rate = header.sample_rate * header.block_align;
        header.data_size = static_cast<uint32_t>(samples.size() * sizeof(float));
        header.file_size = header.data_size + sizeof(WavHeader) - 8;

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        
        // Write raw float data
        file.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(float));
        
        return file.good();
    }

    static bool read(const std::string& path, std::vector<float>& samples, uint32_t& sample_rate) {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;

        WavHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0) {
            return false;
        }

        sample_rate = header.sample_rate;
        size_t num_samples = header.data_size / sizeof(float);
        samples.resize(num_samples);

        file.read(reinterpret_cast<char*>(samples.data()), header.data_size);
        
        return file.good();
    }
};

} // namespace nass_x
