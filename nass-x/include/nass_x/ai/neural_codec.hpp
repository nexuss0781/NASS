#pragma once

#include "neural_engine.hpp"
#include "nass_x/tensor/tensor.hpp"
#include <memory>
#include <string>

namespace nass_x::ai {

/**
 * @brief Neural Audio Codec for latent space compression.
 * 
 * Encodes audio into a compact latent representation and decodes it back.
 * Useful for bandwidth-efficient transmission or storage.
 * Supports neural codecs like Encodec, DAC, or custom VQ-VAE models.
 */
class NeuralCodec {
public:
    explicit NeuralCodec(std::shared_ptr<NeuralEngine> engine);
    
    /**
     * @brief Encode audio into latent representation.
     * @param input Audio tensor [Batch, Channels, Samples]
     * @return Latent codes (quantized indices or continuous vectors)
     */
    Tensor encode(const Tensor& input);
    
    /**
     * @brief Decode latent representation back to audio.
     * @param latents Latent codes from encode()
     * @return Reconstructed audio tensor [Batch, Channels, Samples]
     */
    Tensor decode(const Tensor& latents);
    
    /**
     * @brief Get compression ratio of current codec model.
     * @return Ratio of input samples to latent dimension size
     */
    float getCompressionRatio() const;
    
    /**
     * @brief Get bitrate in bits per second for given sample rate.
     * @param sample_rate Audio sample rate in Hz
     * @return Estimated bitrate
     */
    int getBitrate(int sample_rate) const;

private:
    std::shared_ptr<NeuralEngine> engine_;
    int latent_dim_;
    int hop_length_;
    int codebook_size_;
};

} // namespace nass_x::tensor_x::ai
