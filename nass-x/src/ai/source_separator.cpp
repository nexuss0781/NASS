#include "nass_x/ai/source_separator.hpp"
#include <stdexcept>

namespace nass_x::ai {

SourceSeparator::SourceSeparator(std::shared_ptr<NeuralEngine> engine) 
    : engine_(std::move(engine)) {
    
    if (!engine_) {
        throw std::invalid_argument("NeuralEngine cannot be null");
    }

    // Initialize stem mapping based on model output labels
    // In a real implementation, this would query the model metadata
    stem_map_ = {
        {"vocals", StemType::VOCALS},
        {"drums", StemType::DRUMS},
        {"bass", StemType::BASS},
        {"other", StemType::OTHER}
    };
}

std::unordered_map<SourceSeparator::StemType, Tensor> SourceSeparator::separate(
    const Tensor& input, 
    const std::vector<StemType>& stems) {
    
    std::unordered_map<StemType, Tensor> results;
    
    // 1. Convert time-domain audio to frequency domain (STFT)
    // Note: In production, this would use the shared STFT engine from Phase 3
    // For now, we assume input is already in appropriate format or model handles it
    
    // 2. Run inference to get masks or direct stems
    // Model expected input: [Batch, Channels, Freq, Time] or [Batch, Samples]
    // Model expected output: [Batch, Stems, Channels, Freq, Time] or separate outputs
    
    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> outputs;
    
    if (!engine_->infer(inputs, outputs)) {
        throw std::runtime_error("Neural inference failed");
    }
    
    // 3. Extract requested stems from model output
    // This is a simplified logic; real implementation depends on specific model architecture
    size_t output_idx = 0;
    for (const auto& stem : stems) {
        if (output_idx >= outputs.size()) {
            throw std::runtime_error("Model produced fewer outputs than requested stems");
        }
        
        // Apply masking if model outputs masks, or take direct output if model outputs waveforms
        Tensor stem_tensor = outputs[output_idx];
        
        // Optional: iSTFT if model outputs spectrograms
        // stem_tensor = istft_engine.transform(stem_tensor);
        
        results[stem] = std::move(stem_tensor);
        output_idx++;
    }
    
    return results;
}

size_t SourceSeparator::getStemCount() const {
    // Return number of stems the loaded model supports
    // In real impl, query engine_->getOutputCount() or model metadata
    return stem_map_.size();
}

Tensor SourceSeparator::apply_mask(const Tensor& spec, const Tensor& mask) {
    // Element-wise multiplication: Spectrogram * Mask
    // Requires broadcasting support if mask is single-channel
    if (spec.shape() != mask.shape()) {
        // Handle broadcasting logic here (simplified for now)
        // Real implementation uses SIMD/GPU kernels for efficiency
    }
    
    // Pseudo-code for element-wise multiply
    // return spec * mask; 
    // Actual implementation would use Tensor methods or custom kernel
    
    return mask; // Placeholder
}

} // namespace nass_x::tensor_x::ai
