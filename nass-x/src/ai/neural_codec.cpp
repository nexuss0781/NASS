#include "neural_codec.hpp"
#include <stdexcept>
#include <cmath>

namespace nass_x::ai {

NeuralCodec::NeuralCodec(std::shared_ptr<NeuralEngine> engine) 
    : engine_(std::move(engine)), latent_dim_(128), hop_length_(320), codebook_size_(1024) {
    
    if (!engine_) {
        throw std::invalid_argument("NeuralEngine cannot be null");
    }

    // In real implementation, query model config for these values
    // latent_dim_ = engine_->getConfig().latent_dim;
    // hop_length_ = engine_->getConfig().hop_length;
    // codebook_size_ = engine_->getConfig().codebook_size;
}

Tensor NeuralCodec::encode(const Tensor& input) {
    // Run encoder network
    // Input: [Batch, Channels, Samples]
    // Output: [Batch, LatentDim, CompressedTime] or quantized indices
    
    auto outputs = engine_->infer({input});
    
    if (outputs.empty()) {
        throw std::runtime_error("Encoder produced no output");
    }
    
    // If model uses vector quantization, output might be discrete codes
    // Otherwise it's continuous latent vectors
    return std::move(outputs[0]);
}

Tensor NeuralCodec::decode(const Tensor& latents) {
    // Run decoder network
    // Input: [Batch, LatentDim, CompressedTime] or quantized indices
    // Output: [Batch, Channels, Samples]
    
    auto outputs = engine_->infer({latents});
    
    if (outputs.empty()) {
        throw std::runtime_error("Decoder produced no output");
    }
    
    return std::move(outputs[0]);
}

float NeuralCodec::getCompressionRatio() const {
    // Ratio = InputSamples / LatentSamples
    // Example: 48000 samples/sec -> 150 latents/sec = 320x compression
    return static_cast<float>(hop_length_);
}

int NeuralCodec::getBitrate(int sample_rate) const {
    // Bitrate = (LatentsPerSec * log2(CodebookSize)) bits/sec
    // For continuous latents: LatentsPerSec * LatentDim * 32 (float32)
    
    int latents_per_sec = sample_rate / hop_length_;
    
    // Assuming VQ with codebook (discrete)
    float bits_per_latent = std::log2(static_cast<float>(codebook_size_));
    return static_cast<int>(latents_per_sec * bits_per_latent);
    
    // For continuous:
    // return latents_per_sec * latent_dim_ * 32;
}

} // namespace nass_x::tensor_x::ai
