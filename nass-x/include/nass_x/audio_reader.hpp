/**
 * @file audio_reader.hpp
 * @brief Phase 2: High-performance audio file reader using FFmpeg libav*
 */

#pragma once

#include "nass_x/tensor/tensor.hpp"
#include "nass/memory/arena.hpp"
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <thread>
#include <future>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace nass_x {

/**
 * @brief Audio format information
 */
struct AudioFormat {
    int sample_rate = 0;
    int channels = 0;
    AVSampleFormat sample_format = AV_SAMPLE_FMT_NONE;
    int64_t duration_samples = 0;
    double duration_seconds = 0.0;
    
    std::string format_name() const {
        switch (sample_format) {
            case AV_SAMPLE_FMT_FLT: return "FLT32";
            case AV_SAMPLE_FMT_DBL: return "FLT64";
            case AV_SAMPLE_FMT_S16: return "INT16";
            case AV_SAMPLE_FMT_S32: return "INT32";
            case AV_SAMPLE_FMT_FLTP: return "FLT32P";
            default: return "UNKNOWN";
        }
    }
};

/**
 * @brief Decoded audio frame buffer with move semantics
 */
struct AudioFrame {
    std::vector<float*> planes;  // One per channel
    int nb_samples = 0;
    int64_t pts = 0;  // Presentation timestamp
    bool is_eof = false;
    
    AudioFrame() = default;
    
    // Move constructor
    AudioFrame(AudioFrame&& other) noexcept 
        : planes(std::move(other.planes))
        , nb_samples(other.nb_samples)
        , pts(other.pts)
        , is_eof(other.is_eof) {}
    
    // Move assignment
    AudioFrame& operator=(AudioFrame&& other) noexcept {
        if (this != &other) {
            planes = std::move(other.planes);
            nb_samples = other.nb_samples;
            pts = other.pts;
            is_eof = other.is_eof;
        }
        return *this;
    }
    
    // Delete copy operations
    AudioFrame(const AudioFrame&) = delete;
    AudioFrame& operator=(const AudioFrame&) = delete;
};

/**
 * @brief Thread-safe frame queue for async reading
 */
class FrameQueueReader {
public:
    explicit FrameQueueReader(size_t max_size = 128);
    ~FrameQueueReader();
    
    bool push(AudioFrame&& frame);
    bool pop(AudioFrame& frame, int timeout_ms = -1);
    void set_eof();
    bool is_eof() const { return eof_ && queue_.empty(); }
    size_t size() const;
    void clear();
    
private:
    std::queue<AudioFrame> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    bool eof_;
};

/**
 * @brief High-performance audio reader with multi-threaded decoding
 * 
 * Features:
 * - Native FFmpeg integration (no subprocess pipes)
 * - Multi-threaded async decoding
 * - Hardware acceleration support (NVENC, VAAPI)
 * - Frame-accurate seeking
 * - Dynamic buffer management
 * - Zero-copy where possible
 */
class AudioReader {
public:
    using FrameCallback = std::function<void(AudioFrame&&)>;
    
    explicit AudioReader(const std::string& filepath);
    ~AudioReader();
    
    // Non-copyable, movable
    AudioReader(const AudioReader&) = delete;
    AudioReader& operator=(const AudioReader&) = delete;
    AudioReader(AudioReader&&) noexcept;
    AudioReader& operator=(AudioReader&&) noexcept;
    
    /**
     * @brief Open and probe the audio file
     * @return true if successful
     */
    bool open();
    
    /**
     * @brief Get audio format information
     */
    const AudioFormat& format() const { return format_; }
    
    /**
     * @brief Read next frame (blocking)
     * @param frame Output frame
     * @return false on EOF or error
     */
    bool read_frame(AudioFrame& frame);
    
    /**
     * @brief Read all frames asynchronously with callback
     * @param callback Function called for each decoded frame
     * @param num_threads Number of decoding threads
     */
    void read_all_async(FrameCallback callback, int num_threads = 4);
    
    /**
     * @brief Start background decoding thread
     * @param queue Queue to push decoded frames to
     * @param num_threads Number of decoding threads
     */
    void start_background_decode(FrameQueueReader& queue, int num_threads = 2);
    
    /**
     * @brief Seek to specific sample position
     * @param sample Sample index
     * @return true if successful
     */
    bool seek(int64_t sample);
    
    /**
     * @brief Get current read position in samples
     */
    int64_t position() const { return position_; }
    
    /**
     * @brief Check if end of file reached
     */
    bool eof() const { return eof_; }
    
    /**
     * @brief Close the file and release resources
     */
    void close();
    
    /**
     * @brief Enable hardware acceleration
     * @param device_type HW device type (e.g., "cuda", "vaapi")
     * @return true if successful
     */
    bool enable_hw_accel(const std::string& device_type);
    
private:
    bool init_decoder();
    bool convert_to_float(AVFrame* input, AudioFrame& output);
    void decode_thread(FrameCallback callback, int thread_id, int total_threads);
    void background_decode_thread(FrameQueueReader& queue, int thread_id);
    bool init_hw_device(const std::string& device_type);
    
    std::string filepath_;
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    SwrContext* swr_ctx_;
    AVAudioFifo* fifo_;
    AVBufferRef* hw_device_ctx_;
    
    AudioFormat format_;
    int stream_index_;
    int64_t position_;
    bool eof_;
    bool opened_;
    bool hw_accel_enabled_;
    
    std::unique_ptr<nass_x::tensor::Arena> arena_;
    std::vector<std::thread> decode_threads_;
};

} // namespace nass_x::tensor_x
