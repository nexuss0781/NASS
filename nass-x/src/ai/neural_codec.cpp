#include "nass_x/ai/neural_codec.hpp"
#include <vector>
#include <cmath>

namespace nass_x::ai {

NeuralCodec::NeuralCodec(const std::string& model_path, const Config& config)
    : config_(config), codebook_size_(config.codebook_size), latent_dim_(config.latent_dim) {
    (void)model_path;
    codebook_.resize(codebook_size_ * latent_dim_);
    for (size_t i = 0; i < codebook_.size(); ++i) {
        codebook_[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

Tensor NeuralCodec::encode(const Tensor& audio) {
    size_t batch = audio.shape[0];
    size_t time = audio.shape[1];
    size_t latent_time = time / config_.downsample_factor;
    
    Tensor latent(batch, latent_time, latent_dim_);
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < latent_time; ++t) {
            size_t src_idx = t * config_.downsample_factor;
            if (src_idx < time) {
                float val = audio.data[b * time + src_idx];
                int code_idx = static_cast<int>(std::abs(val) * (codebook_size_ - 1)) % codebook_size_;
                latent.data[b * latent_time * latent_dim_ + t * latent_dim_] = 
                    codebook_[code_idx * latent_dim_];
            }
        }
    }
    
    return latent;
}

Tensor NeuralCodec::decode(const Tensor& latent) {
    size_t batch = latent.shape[0];
    size_t latent_time = latent.shape[1];
    size_t out_time = latent_time * config_.downsample_factor;
    
    Tensor audio(batch, out_time);
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < latent_time; ++t) {
            float val = latent.data[b * latent_time * latent_dim_ + t * latent_dim_];
            for (int i = 0; i < config_.downsample_factor && (t * config_.downsample_factor + i) < static_cast<int>(out_time); ++i) {
                audio.data[b * out_time + t * config_.downsample_factor + i] = val;
            }
        }
    }
    
    return audio;
}

float NeuralCodec::compression_ratio() const {
    float original_bits = 16.0f;
    float compressed_bits = std::log2(static_cast<float>(codebook_size_)) / config_.downsample_factor;
    return original_bits / compressed_bits;
}

} // namespace nass_x::ai
