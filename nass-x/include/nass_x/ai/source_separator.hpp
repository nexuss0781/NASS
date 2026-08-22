#pragma once

#include "nass_x/ai/neural_engine.hpp"
#include "nass_x/tensor/tensor.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace nass_x::ai {

/**
 * @brief Real-time source separation module using spectral masking.
 */
class SourceSeparator {
public:
    struct Config {
        size_t window_size = 2048;
        size_t hop_size = 512;
    };

    explicit SourceSeparator(const std::string& model_path, const Config& config = Config());

    enum class SourceType {
        VOCALS,
        INSTRUMENTAL,
        DRUMS,
        BASS,
        OTHER
    };

    Tensor separate(const Tensor& input_audio, SourceType target);

private:
    Config config_;
    size_t window_size_;
    size_t hop_size_;
    std::vector<float> hann_window_;

    Tensor generate_mask(const Tensor& spec, SourceType target);
    Tensor stft(const Tensor& audio, int window_size, int hop_size);
    Tensor istft(const Tensor& spec, int window_size, int hop_size, const std::vector<float>& window);
};

} // namespace nass_x::ai
