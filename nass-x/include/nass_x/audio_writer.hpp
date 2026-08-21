/**
 * @file audio_writer.hpp
 * @brief High-performance audio file writer using FFmpeg libav*
 */

#pragma once

#include "nass/tensor/tensor.hpp"
#include <string>
#include <memory>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace nass_x {

/**
 * @brief Audio encoding parameters
 */
struct EncodeParams {
    int sample_rate = 48000;
    int channels = 2;
    int bit_rate = 320000;  // bits per second
    AVSampleFormat sample_format = AV_SAMPLE_FMT_FLTP;
    AVCodecID codec_id = AV_CODEC_ID_AAC;
    const char* output_format = "mp4";
    
    // Multi-quality encoding
    std::vector<std::tuple<int, int, int>> quality_profiles;  // (sample_rate, channels, bit_rate)
};

/**
 * @brief High-performance audio encoder with multi-threaded encoding
 */
class AudioWriter {
public:
    explicit AudioWriter(const std::string& filepath, const EncodeParams& params = EncodeParams());
    ~AudioWriter();
    
    // Non-copyable, movable
    AudioWriter(const AudioWriter&) = delete;
    AudioWriter& operator=(const AudioWriter&) = delete;
    AudioWriter(AudioWriter&&) noexcept;
    AudioWriter& operator=(AudioWriter&&) noexcept;
    
    /**
     * @brief Open and initialize the encoder
     * @return true if successful
     */
    bool open();
    
    /**
     * @brief Write audio samples (interleaved float format)
     * @param samples Pointer to sample buffer
     * @param num_samples Number of samples per channel
     * @return true if successful
     */
    bool write_samples(const float* samples, int num_samples);
    
    /**
     * @brief Write audio samples from tensor
     * @param tensor Audio tensor [channels, samples]
     * @return true if successful
     */
    bool write_tensor(const Tensor<float>& tensor);
    
    /**
     * @brief Flush remaining frames and close
     * @return true if successful
     */
    bool finalize();
    
    /**
     * @brief Get number of samples written
     */
    int64_t samples_written() const { return samples_written_; }
    
    /**
     * @brief Enable multi-quality simultaneous encoding
     * @param profiles Vector of (sample_rate, channels, bit_rate) tuples
     */
    void enable_multi_quality(std::vector<std::tuple<int, int, int>> profiles);
    
private:
    bool init_encoder();
    bool encode_and_mux(AVFrame* frame);
    bool resample_if_needed(AVFrame* input, AVFrame*& output);
    
    std::string filepath_;
    EncodeParams params_;
    
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    SwrContext* swr_ctx_;
    AVAudioFifo* fifo_;
    
    int stream_index_;
    int64_t samples_written_;
    int64_t next_pts_;
    bool opened_;
    
    // Multi-quality encoders
    struct QualityEncoder {
        AVFormatContext* format_ctx;
        AVCodecContext* codec_ctx;
        SwrContext* swr_ctx;
        int stream_index;
        int64_t samples_written;
    };
    std::vector<QualityEncoder> quality_encoders_;
};

} // namespace nass_x
