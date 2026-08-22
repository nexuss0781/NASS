#include <fstream>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>

struct WavHeader {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};

int main() {
    const int duration_sec = 2;
    const int sample_rate = 44100;
    const float freq = 440.0f;
    const int num_samples = duration_sec * sample_rate;
    
    std::vector<float> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        samples[i] = 0.5f * sinf(2.0f * 3.14159265358979323846f * freq * t);
    }

    WavHeader header{};
    std::memcpy(header.riff, "RIFF", 4);
    std::memcpy(header.wave, "WAVE", 4);
    std::memcpy(header.fmt, "fmt ", 4);
    std::memcpy(header.data, "data", 4);
    
    header.fmt_size = 16;
    header.audio_format = 1; // PCM
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 32; // Float32
    header.byte_rate = sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = num_samples * sizeof(float);
    header.file_size = 36 + header.data_size;

    std::ofstream out("/workspace/nass-x/test_input.wav", std::ios::binary);
    if (!out) {
        std::cerr << "Failed to create file\n";
        return 1;
    }

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(samples.data()), sizeof(float) * num_samples);
    out.close();

    std::cout << "Generated test_input.wav (" << num_samples << " samples, 32-bit float)\n";
    return 0;
}
