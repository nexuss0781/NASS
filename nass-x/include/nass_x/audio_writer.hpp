/**
 * @file audio_writer.hpp
 * @brief Phase 2: High-performance audio file writer using FFmpeg libav*
 */

#pragma once

#include "nass_x/tensor/tensor.hpp"
#include "nass/memory/arena.hpp"
#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace nass_x {

/**
 * @brief Audio output format configuration
 */
struct AudioOutputFormat {
    int sample_rate = 48000;
    int channels = 2;
    AVSampleFormat sample_format = AV_SAMPLE_FMT_FLTP;
    const char* codec_name = "pcm_f32le";
    const char* container_format = "wav";
    int bitrate = 0;  // 0 for PCM
};

/**
 * @brief Thread-safe frame queue for async writing
 */
class FrameQueueWriter {
public:
    explicit FrameQueueWriter(size_t max_size = 128);
    ~FrameQueueWriter();
    
    bool push(std::vector<float*>&& planes, int nb_samples);
    bool pop(std::vector<float*>& planes, int& nb_samples, int timeout_ms = -1);
    void set_eof();
    bool is_eof() const { return eof_ && queue_.empty(); }
    size_t size() const;
    void clear();
    
private:
    struct FrameData {
        std::vector<float*> planes;
        int nb_samples;
    };
    
    std::queue<FrameData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    bool eof_;
};

/**
 * @brief High-performance audio writer with multi-threaded encoding
 * 
 * Features:
 * - Native FFmpeg integration
 * - Multi-threaded async encoding
 * - Hardware acceleration support
 * - Dynamic buffer management
 * - Real-time streaming capable
 */
class AudioWriter {
public:
    explicit AudioWriter(const std::string& filepath, 
                        const AudioOutputFormat& format = AudioOutputFormat());
    ~AudioWriter();
    
    // Non-copyable, movable
    AudioWriter(const AudioWriter&) = delete;
    AudioWriter& operator=(const AudioWriter&) = delete;
    AudioWriter(AudioWriter&&) noexcept;
    AudioWriter& operator=(AudioWriter&&) noexcept;
    
    /**
     * @brief Open and initialize the output file
     * @return true if successful
     */
    bool open();
    
    /**
     * @brief Write audio frame (blocking)
     * @param planes Channel data pointers
     * @param nb_samples Number of samples per channel
     * @return true if successful
     */
    bool write_frame(const std::vector<float*>& planes, int nb_samples);
    
    /**
     * @brief Write audio frame asynchronously
     * @param planes Channel data pointers (will be copied)
     * @param nb_samples Number of samples per channel
     * @return true if queued successfully
     */
    bool write_frame_async(std::vector<float*>&& planes, int nb_samples);
    
    /**
     * @brief Flush all pending frames and close
     */
    void close();
    
    /**
     * @brief Get number of frames written
     */
    uint64_t frames_written() const { return frames_written_; }
    
    /**
     * @brief Get total samples written
     */
    uint64_t samples_written() const { return samples_written_; }
    
    /**
     * @brief Check if writer is open
     */
    bool is_open() const { return opened_; }
    
    /**
     * @brief Enable hardware acceleration
     * @param device_type HW device type (e.g., "cuda", "vaapi")
     * @return true if successful
     */
    bool enable_hw_accel(const std::string& device_type);
    
private:
    bool init_encoder();
    bool init_resampler();
    bool convert_from_float(const std::vector<float*>& input, int nb_samples, 
                           std::vector<uint8_t*>& output);
    bool encode_and_write(AVFrame* frame);
    void encode_thread();
    bool init_hw_device(const std::string& device_type);
    
    std::string filepath_;
    AudioOutputFormat format_;
    
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    SwrContext* swr_ctx_;
    AVAudioFifo* fifo_;
    AVBufferRef* hw_device_ctx_;
    
    int stream_index_;
    bool opened_;
    bool encoder_opened_;
    bool hw_accel_enabled_;
    
    std::unique_ptr<FrameQueueWriter> frame_queue_;
    std::thread encode_thread_;
    std::atomic<bool> stop_encode_;
    
    std::unique_ptr<nass_x::tensor::Arena> arena_;
    uint64_t frames_written_;
    uint64_t samples_written_;
    int64_t pts_;
};

} // namespace nass_x::tensor_x
