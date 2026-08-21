/**
 * @file audio_reader.cpp
 * @brief Implementation of high-performance audio reader
 */

#include "nass_x/audio_reader.hpp"
#include <iostream>
#include <cstring>
#include <thread>
#include <future>

namespace nass_x {

AudioReader::AudioReader(const std::string& filepath)
    : filepath_(filepath)
    , format_ctx_(nullptr)
    , codec_ctx_(nullptr)
    , swr_ctx_(nullptr)
    , fifo_(nullptr)
    , stream_index_(-1)
    , position_(0)
    , eof_(false)
    , opened_(false)
    , arena_(std::make_unique<nass::Arena>(16 * 1024 * 1024))
{
    av_log_set_level(AV_LOG_ERROR);
}

AudioReader::~AudioReader() {
    close();
}

AudioReader::AudioReader(AudioReader&& other) noexcept
    : filepath_(std::move(other.filepath_))
    , format_ctx_(other.format_ctx_)
    , codec_ctx_(other.codec_ctx_)
    , swr_ctx_(other.swr_ctx_)
    , fifo_(other.fifo_)
    , format_(other.format_)
    , stream_index_(other.stream_index_)
    , position_(other.position_)
    , eof_(other.eof_)
    , opened_(other.opened_)
    , arena_(std::move(other.arena_))
{
    other.format_ctx_ = nullptr;
    other.codec_ctx_ = nullptr;
    other.swr_ctx_ = nullptr;
    other.fifo_ = nullptr;
    other.stream_index_ = -1;
    other.opened_ = false;
}

AudioReader& AudioReader::operator=(AudioReader&& other) noexcept {
    if (this != &other) {
        close();
        filepath_ = std::move(other.filepath_);
        format_ctx_ = other.format_ctx_;
        codec_ctx_ = other.codec_ctx_;
        swr_ctx_ = other.swr_ctx_;
        fifo_ = other.fifo_;
        format_ = other.format_;
        stream_index_ = other.stream_index_;
        position_ = other.position_;
        eof_ = other.eof_;
        opened_ = other.opened_;
        arena_ = std::move(other.arena_);
        
        other.format_ctx_ = nullptr;
        other.codec_ctx_ = nullptr;
        other.swr_ctx_ = nullptr;
        other.fifo_ = nullptr;
        other.stream_index_ = -1;
        other.opened_ = false;
    }
    return *this;
}

bool AudioReader::open() {
    if (opened_) return true;
    
    int ret = avformat_open_input(&format_ctx_, filepath_.c_str(), nullptr, nullptr);
    if (ret < 0) {
        std::cerr << "Error opening file: " << filepath_ << std::endl;
        return false;
    }
    
    ret = avformat_find_stream_info(format_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "Error finding stream info" << std::endl;
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    stream_index_ = av_find_best_stream(format_ctx_, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_index_ < 0) {
        std::cerr << "No audio stream found" << std::endl;
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    AVCodecParameters* codec_params = format_ctx_->streams[stream_index_]->codecpar;
    
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        std::cerr << "Codec not found" << std::endl;
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        std::cerr << "Error allocating codec context" << std::endl;
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    ret = avcodec_parameters_to_context(codec_ctx_, codec_params);
    if (ret < 0) {
        std::cerr << "Error copying codec parameters" << std::endl;
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    codec_ctx_->thread_count = std::thread::hardware_concurrency();
    codec_ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    
    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        std::cerr << "Error opening codec" << std::endl;
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    format_.sample_rate = codec_ctx_->sample_rate;
    format_.channels = codec_ctx_->ch_layout.nb_channels;
    format_.sample_format = codec_ctx_->sample_fmt;
    
    int64_t duration = format_ctx_->duration;
    if (duration != AV_NOPTS_VALUE) {
        format_.duration_seconds = duration / (double)AV_TIME_BASE;
        format_.duration_samples = static_cast<int64_t>(format_.duration_seconds * format_.sample_rate);
    } else {
        format_.duration_seconds = 0;
        format_.duration_samples = 0;
    }
    
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
        std::cerr << "Error allocating resampler" << std::endl;
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    av_opt_set_chlayout(swr_ctx_, "in_chlayout", &codec_ctx_->ch_layout, 0);
    av_opt_set_int(swr_ctx_, "in_sample_rate", codec_ctx_->sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "in_sample_fmt", codec_ctx_->sample_fmt, 0);
    
    av_opt_set_chlayout(swr_ctx_, "out_chlayout", &codec_ctx_->ch_layout, 0);
    av_opt_set_int(swr_ctx_, "out_sample_rate", codec_ctx_->sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "out_sample_fmt", AV_SAMPLE_FMT_FLTP, 0);
    
    ret = swr_init(swr_ctx_);
    if (ret < 0) {
        std::cerr << "Error initializing resampler" << std::endl;
        swr_free(&swr_ctx_);
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    fifo_ = av_audio_fifo_alloc(AV_SAMPLE_FMT_FLTP, format_.channels, 1024);
    if (!fifo_) {
        std::cerr << "Error creating audio FIFO" << std::endl;
        swr_free(&swr_ctx_);
        avcodec_free_context(&codec_ctx_);
        avformat_close_input(&format_ctx_);
        return false;
    }
    
    opened_ = true;
    return true;
}

bool AudioReader::read_frame(AudioFrame& frame) {
    if (!opened_) return false;
    if (eof_) return false;
    
    AVPacket* packet = av_packet_alloc();
    AVFrame* decoded_frame = av_frame_alloc();
    
    bool got_frame = false;
    
    while (!got_frame) {
        if (av_audio_fifo_size(fifo_) >= codec_ctx_->frame_size) {
            frame.nb_samples = codec_ctx_->frame_size;
            frame.planes.resize(format_.channels);
            
            for (int c = 0; c < format_.channels; c++) {
                frame.planes[c] = static_cast<float*>(
                    arena_->allocate(frame.nb_samples * sizeof(float)));
            }
            
            void** data = reinterpret_cast<void**>(frame.planes.data());
            int ret = av_audio_fifo_read(fifo_, data, frame.nb_samples);
            if (ret > 0) {
                frame.pts = position_;
                position_ += ret;
                got_frame = true;
            }
            break;
        }
        
        int ret = av_read_frame(format_ctx_, packet);
        if (ret < 0) {
            if (ret == AVERROR_EOF) {
                eof_ = true;
                if (av_audio_fifo_size(fifo_) > 0) {
                    frame.nb_samples = av_audio_fifo_size(fifo_);
                    frame.planes.resize(format_.channels);
                    
                    for (int c = 0; c < format_.channels; c++) {
                        frame.planes[c] = static_cast<float*>(
                            arena_->allocate(frame.nb_samples * sizeof(float)));
                    }
                    
                    void** data = reinterpret_cast<void**>(frame.planes.data());
                    av_audio_fifo_read(fifo_, data, frame.nb_samples);
                    frame.pts = position_;
                    position_ += frame.nb_samples;
                    got_frame = true;
                }
            }
            break;
        }
        
        if (packet->stream_index != stream_index_) {
            av_packet_unref(packet);
            continue;
        }
        
        ret = avcodec_send_packet(codec_ctx_, packet);
        av_packet_unref(packet);
        
        if (ret < 0) {
            std::cerr << "Error sending packet to decoder" << std::endl;
            break;
        }
        
        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, decoded_frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                std::cerr << "Error decoding frame" << std::endl;
                break;
            }
            
            AudioFrame temp_frame;
            if (convert_to_float(decoded_frame, temp_frame)) {
                av_audio_fifo_write(fifo_, reinterpret_cast<void**>(temp_frame.planes.data()), 
                                   temp_frame.nb_samples);
            }
        }
    }
    
    av_frame_free(&decoded_frame);
    av_packet_free(&packet);
    
    return got_frame;
}

bool AudioReader::convert_to_float(AVFrame* input, AudioFrame& output) {
    output.nb_samples = input->nb_samples;
    output.planes.resize(format_.channels);
    
    for (int c = 0; c < format_.channels; c++) {
        output.planes[c] = static_cast<float*>(
            arena_->allocate(output.nb_samples * sizeof(float)));
    }
    
    void** data = reinterpret_cast<void**>(output.planes.data());
    int ret = swr_convert(swr_ctx_, data, output.nb_samples, 
                         (const uint8_t**)input->extended_data, input->nb_samples);
    
    return ret > 0;
}

void AudioReader::read_all_async(FrameCallback callback, int num_threads) {
    if (!opened_) return;
    
    std::vector<std::future<void>> futures;
    
    for (int t = 0; t < num_threads; t++) {
        futures.push_back(std::async(std::launch::async, 
            [this, callback, t, num_threads]() {
                decode_thread(callback, t, num_threads);
            }));
    }
    
    for (auto& f : futures) {
        f.get();
    }
}

void AudioReader::decode_thread(FrameCallback callback, int /*thread_id*/, int /*total_threads*/) {
    AudioFrame frame;
    while (read_frame(frame)) {
        if (!frame.is_eof) {
            callback(std::move(frame));
        }
    }
}

bool AudioReader::seek(int64_t sample) {
    if (!opened_) return false;
    
    int64_t timestamp = av_rescale_q(sample, {1, format_.sample_rate}, 
                                     format_ctx_->streams[stream_index_]->time_base);
    
    int ret = av_seek_frame(format_ctx_, stream_index_, timestamp, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) return false;
    
    avcodec_flush_buffers(codec_ctx_);
    av_audio_fifo_reset(fifo_);
    
    position_ = sample;
    eof_ = false;
    
    return true;
}

void AudioReader::close() {
    if (fifo_) {
        av_audio_fifo_free(fifo_);
        fifo_ = nullptr;
    }
    if (swr_ctx_) {
        swr_free(&swr_ctx_);
        swr_ctx_ = nullptr;
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
        format_ctx_ = nullptr;
    }
    opened_ = false;
}

} // namespace nass_x
