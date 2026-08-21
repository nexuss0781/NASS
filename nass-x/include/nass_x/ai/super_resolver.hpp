#pragma once

#include "neural_engine.hpp"
#include "nass_x/tensor/tensor.hpp"
#include <memory>

namespace nass_x::ai {

/**
 * @brief Neural Super-Resolution for bandwidth extension.
 * 
 * Upsamples low-bandwidth audio (e.g., 8kHz telephone quality) to high-fidelity 
 * wideband audio (e.g., 48kHz) using deep learning models.
 * Reconstructs missing high-frequency content harmonically and texturally.
 */
class SuperResolver {
public:
    explicit SuperResolver(std::shared_ptr<NeuralEngine> engine);
    
    /**
     * @brief Upsample audio from low to high sample rate.
     * @param input Low-bandwidth audio tensor [Batch, Channels, LowSamples]
     * @param target_sample_rate Desired output sample rate (e.g., 48000)
     * @return High-fidelity upsampled audio tensor [Batch, Channels, HighSamples]
     */
    Tensor upsample(const Tensor& input, int target_sample_rate);
    
    /**
     * @brief Get supported input sample rates.
     * @return Vector of supported low-band sample rates (e.g., {8000, 16000})
     */
    std::vector<int> getSupportedInputRates() const;
    
    /**
     * @brief Get maximum upsampling factor.
     * @return Max ratio (e.g., 6x for 8kHz->48kHz)
     */
    int getMaxUpsamplingFactor() const;

private:
    std::shared_ptr<NeuralEngine> engine_;
    std::vector<int> supported_input_rates_;
    int max_upsample_factor_;
};

} // namespace nass_x::tensor_x::ai
