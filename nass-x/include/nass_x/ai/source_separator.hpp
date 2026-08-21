#pragma once

#include "neural_engine.hpp"
#include "nass_x/tensor/tensor.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace nass_x::ai {

/**
 * @brief Real-time source separation module using spectral masking.
 * 
 * Takes a mixed audio tensor and outputs separated stems (e.g., Vocals, Drums, Bass, Other).
 * Uses a U-Net or Masking architecture via the underlying NeuralEngine.
 */
class SourceSeparator {
public:
    enum class StemType {
        VOCALS,
        DRUMS,
        BASS,
        OTHER,
        FULL_BAND
    };

    explicit SourceSeparator(std::shared_ptr<NeuralEngine> engine);
    
    /**
     * @brief Separate audio into specified stems.
     * @param input Mixed audio tensor [Batch, Channels, Time]
     * @param stems List of desired stem types to extract
     * @return Map of StemType to separated audio tensors
     */
    std::unordered_map<StemType, Tensor> separate(
        const Tensor& input, 
        const std::vector<StemType>& stems);

    /**
     * @brief Get number of available stems from current model.
     */
    size_t getStemCount() const;

private:
    std::shared_ptr<NeuralEngine> engine_;
    std::unordered_map<std::string, StemType> stem_map_;
    
    // Apply soft mask to spectrogram
    Tensor apply_mask(const Tensor& spec, const Tensor& mask);
};

} // namespace nass_x::tensor_x::ai
