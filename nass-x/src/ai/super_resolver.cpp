#include "super_resolver.hpp"
#include <stdexcept>
#include <algorithm>

namespace nass_x::ai {

SuperResolver::SuperResolver(std::shared_ptr<NeuralEngine> engine) 
    : engine_(std::move(engine)), 
      supported_input_rates_({8000, 16000, 24000}),
      max_upsample_factor_(6) {
    
    if (!engine_) {
        throw std::invalid_argument("NeuralEngine cannot be null");
    }

    // In real implementation, query model config
    // supported_input_rates_ = engine_->getConfig().input_rates;
    // max_upsample_factor_ = engine_->getConfig().max_factor;
}

Tensor SuperResolver::upsample(const Tensor& input, int target_sample_rate) {
    // Get input sample rate from metadata or infer from shape
    // For this example, assume we know it or model handles it internally
    
    // Validate target rate
    int input_rate = 8000; // Placeholder - would be determined from context
    int factor = target_sample_rate / input_rate;
    
    if (factor > max_upsample_factor_) {
        throw std::invalid_argument(
            "Upsampling factor " + std::to_string(factor) + 
            " exceeds maximum supported factor " + std::to_string(max_upsample_factor_));
    }
    
    // Run super-resolution model
    // Input: [Batch, Channels, LowSamples]
    // Output: [Batch, Channels, HighSamples] where HighSamples = LowSamples * factor
    
    auto outputs = engine_->infer({input});
    
    if (outputs.empty()) {
        throw std::runtime_error("Super-resolution model produced no output");
    }
    
    return std::move(outputs[0]);
}

std::vector<int> SuperResolver::getSupportedInputRates() const {
    return supported_input_rates_;
}

int SuperResolver::getMaxUpsamplingFactor() const {
    return max_upsample_factor_;
}

} // namespace nass_x::tensor_x::ai
