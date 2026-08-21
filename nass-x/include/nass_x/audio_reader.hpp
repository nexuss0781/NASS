/**
 * @file audio_reader.hpp
 * @brief High-performance audio file reader using FFmpeg libav*
 */

#pragma once

#include "nass/tensor/tensor.hpp"
#include "nass/memory/arena.hpp"
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace nass_x {

// Alias for nass::Arena
using MemoryArena = nass::Arena;

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
    int sample_rate;
    int channels;
    AVSampleFormat sample_format;
    int64_t duration_samples;
    double duration_seconds;
    
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
 * @brief Decoded audio frame buffer
 */
struct AudioFrame {
    std::vector<float*> planes;  // One per channel
    int nb_samples;
    int64_t pts;  // Presentation timestamp
    bool is_eof;
    
    AudioFrame() : nb_samples(0), pts(0), is_eof(false) {}
};

/**
 * @brief High-performance audio reader with multi-threaded decoding
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
    
private:
    bool init_decoder();
    bool convert_to_float(AVFrame* input, AudioFrame& output);
    void decode_thread(FrameCallback callback, int thread_id, int total_threads);
    
    std::string filepath_;
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    SwrContext* swr_ctx_;
    AVAudioFifo* fifo_;
    
    AudioFormat format_;
    int stream_index_;
    int64_t position_;
    bool eof_;
    bool opened_;
    
    std::unique_ptr<nass::Arena> arena_;
};

} // namespace nass_x
