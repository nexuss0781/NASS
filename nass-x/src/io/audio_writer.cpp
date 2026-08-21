/**
 * @file audio_writer.cpp
 * @brief Phase 2: Implementation of high-performance audio writer
 */

#include "nass_x/audio_writer.hpp"
#include <iostream>
#include <cstring>
#include <thread>
#include <sys/stat.h>

namespace nass_x {

// FrameQueueWriter implementation
FrameQueueWriter::FrameQueueWriter(size_t max_size) 
    : max_size_(max_size)
    , eof_(false) {}

FrameQueueWriter::~FrameQueueWriter() {
    clear();
}

bool FrameQueueWriter::push(std::vector<float*>&& planes, int nb_samples) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    cv_not_full_.wait(lock, [this]() { return queue_.size() < max_size_ || eof_; });
    
    if (eof_) return false;
    
    FrameData data;
    data.planes = std::move(planes);
    data.nb_samples = nb_samples;
    queue_.push(std::move(data));
    
    cv_not_empty_.notify_one();
    return true;
}

bool FrameQueueWriter::pop(std::vector<float*>& planes, int& nb_samples, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (timeout_ms < 0) {
        cv_not_empty_.wait(lock, [this]() { return !queue_.empty() || eof_; });
    } else {
        if (!cv_not_empty_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                         [this]() { return !queue_.empty() || eof_; })) {
            return false;
        }
    }
    
    if (queue_.empty()) {
        return eof_;
    }
    
    auto& data = queue_.front();
    planes = std::move(data.planes);
    nb_samples = data.nb_samples;
    queue_.pop();
    
    cv_not_full_.notify_one();
    return true;
}

void FrameQueueWriter::set_eof() {
    std::lock_guard<std::mutex> lock(mutex_);
    eof_ = true;
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
}

size_t FrameQueueWriter::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void FrameQueueWriter::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
    eof_ = false;
}

// AudioWriter implementation
AudioWriter::AudioWriter(const std::string& filepath, 
                        const AudioOutputFormat& format)
    : filepath_(filepath)
    , format_(format)
    , format_ctx_(nullptr)
    , codec_ctx_(nullptr)
    , swr_ctx_(nullptr)
    , fifo_(nullptr)
    , hw_device_ctx_(nullptr)
    , stream_index_(-1)
    , opened_(false)
    , encoder_opened_(false)
    , hw_accel_enabled_(false)
    , frame_queue_(std::make_unique<FrameQueueWriter>(128))
    , stop_encode_(false)
    , arena_(std::make_unique<nass::Arena>(32 * 1024 * 1024))
    , frames_written_(0)
    , samples_written_(0)
    , pts_(0)
{
    av_log_set_level(AV_LOG_ERROR);
}

AudioWriter::~AudioWriter() {
    close();
}

AudioWriter::AudioWriter(AudioWriter&& other) noexcept
    : filepath_(std::move(other.filepath_))
    , format_(other.format_)
    , format_ctx_(other.format_ctx_)
    , codec_ctx_(other.codec_ctx_)
    , swr_ctx_(other.swr_ctx_)
    , fifo_(other.fifo_)
    , hw_device_ctx_(other.hw_device_ctx_)
    , stream_index_(other.stream_index_)
    , opened_(other.opened_)
    , encoder_opened_(other.encoder_opened_)
    , hw_accel_enabled_(other.hw_accel_enabled_)
    , frame_queue_(std::move(other.frame_queue_))
    , encode_thread_(std::move(other.encode_thread_))
    , stop_encode_(other.stop_encode_.load())
    , arena_(std::move(other.arena_))
    , frames_written_(other.frames_written_)
    , samples_written_(other.samples_written_)
    , pts_(other.pts_)
{
    other.format_ctx_ = nullptr;
    other.codec_ctx_ = nullptr;
    other.swr_ctx_ = nullptr;
    other.fifo_ = nullptr;
    other.hw_device_ctx_ = nullptr;
    other.stream_index_ = -1;
    other.opened_ = false;
    other.encoder_opened_ = false;
    other.hw_accel_enabled_ = false;
    other.stop_encode_ = true;
}

AudioWriter& AudioWriter::operator=(AudioWriter&& other) noexcept {
    if (this != &other) {
        close();
        filepath_ = std::move(other.filepath_);
        format_ = other.format_;
        format_ctx_ = other.format_ctx_;
        codec_ctx_ = other.codec_ctx_;
        swr_ctx_ = other.swr_ctx_;
        fifo_ = other.fifo_;
        hw_device_ctx_ = other.hw_device_ctx_;
        stream_index_ = other.stream_index_;
        opened_ = other.opened_;
        encoder_opened_ = other.encoder_opened_;
        hw_accel_enabled_ = other.hw_accel_enabled_;
        frame_queue_ = std::move(other.frame_queue_);
        encode_thread_ = std::move(other.encode_thread_);
        stop_encode_ = other.stop_encode_.load();
        arena_ = std::move(other.arena_);
        frames_written_ = other.frames_written_;
        samples_written_ = other.samples_written_;
        pts_ = other.pts_;
        
        other.format_ctx_ = nullptr;
        other.codec_ctx_ = nullptr;
        other.swr_ctx_ = nullptr;
        other.fifo_ = nullptr;
        other.hw_device_ctx_ = nullptr;
        other.stream_index_ = -1;
        other.opened_ = false;
        other.encoder_opened_ = false;
        other.hw_accel_enabled_ = false;
        other.stop_encode_ = true;
    }
    return *this;
}

bool AudioWriter::enable_hw_accel(const std::string& device_type) {
    return init_hw_device(device_type);
}

bool AudioWriter::init_hw_device(const std::string& device_type) {
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(device_type.c_str());
    if (type == AV_HWDEVICE_TYPE_NONE) {
        std::cerr << "Hardware device type not supported: " << device_type << std::endl;
        return false;
    }
    
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, type, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create hardware device context" << std::endl;
        return false;
    }
    
    hw_accel_enabled_ = true;
    return true;
}

bool AudioWriter::open() {
    if (opened_) return true;
    
    // Create output directory if needed
    size_t last_slash = filepath_.find_last_of('/');
    if (last_slash != std::string::npos) {
        std::string dir = filepath_.substr(0, last_slash);
        if (!dir.empty()) {
            mkdir(dir.c_str(), 0755);
        }
    }
    
    int ret = avformat_alloc_output_context2(&format_ctx_, nullptr, 
                                             format_.container_format, nullptr);
    if (ret < 0 || !format_ctx_) {
        std::cerr << "Error allocating output context" << std::endl;
        return false;
    }
    
    const AVCodec* codec = avcodec_find_encoder_by_name(format_.codec_name);
    if (!codec) {
        std::cerr << "Codec not found: " << format_.codec_name << std::endl;
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    AVStream* stream = avformat_new_stream(format_ctx_, nullptr);
    if (!stream) {
        std::cerr << "Error creating stream" << std::endl;
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    stream_index_ = stream->index;
    
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        std::cerr << "Error allocating codec context" << std::endl;
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    codec_ctx_->sample_rate = format_.sample_rate;
    codec_ctx_->ch_layout.nb_channels = format_.channels;
    codec_ctx_->sample_fmt = format_.sample_format;
    codec_ctx_->bit_rate = format_.bitrate;
    codec_ctx_->time_base = {1, format_.sample_rate};
    
    // Enable hardware acceleration if configured
    if (hw_accel_enabled_ && hw_device_ctx_) {
        codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    }
    
    if (format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    
    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        std::cerr << "Error opening codec" << std::endl;
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx_);
    if (ret < 0) {
        std::cerr << "Error copying codec parameters" << std::endl;
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    // Initialize resampler
    if (!init_resampler()) {
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    // Create FIFO buffer
    fifo_ = av_audio_fifo_alloc(format_.sample_format, format_.channels, 2048);
    if (!fifo_) {
        std::cerr << "Error creating audio FIFO" << std::endl;
        swr_free(&swr_ctx_);
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    // Open output file
    if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx_->pb, filepath_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            std::cerr << "Error opening output file: " << filepath_ << std::endl;
            av_audio_fifo_free(fifo_);
            swr_free(&swr_ctx_);
            avcodec_free_context(&codec_ctx_);
            avformat_free_context(format_ctx_);
            format_ctx_ = nullptr;
            return false;
        }
    }
    
    // Write header
    ret = avformat_write_header(format_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "Error writing header" << std::endl;
        if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx_->pb);
        }
        av_audio_fifo_free(fifo_);
        swr_free(&swr_ctx_);
        avcodec_free_context(&codec_ctx_);
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        return false;
    }
    
    // Start encoder thread
    stop_encode_ = false;
    encode_thread_ = std::thread(&AudioWriter::encode_thread, this);
    
    opened_ = true;
    encoder_opened_ = true;
    return true;
}

bool AudioWriter::init_encoder() {
    return true;  // Already done in open()
}

bool AudioWriter::init_resampler() {
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) return false;
    
    AVChannelLayout ch_layout;
    av_channel_layout_default(&ch_layout, format_.channels);
    
    av_opt_set_chlayout(swr_ctx_, "in_chlayout", &ch_layout, 0);
    av_opt_set_int(swr_ctx_, "in_sample_rate", format_.sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "in_sample_fmt", AV_SAMPLE_FMT_FLTP, 0);
    
    av_opt_set_chlayout(swr_ctx_, "out_chlayout", &ch_layout, 0);
    av_opt_set_int(swr_ctx_, "out_sample_rate", format_.sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "out_sample_fmt", format_.sample_format, 0);
    
    int ret = swr_init(swr_ctx_);
    av_channel_layout_uninit(&ch_layout);
    
    return ret >= 0;
}

bool AudioWriter::write_frame(const std::vector<float*>& planes, int nb_samples) {
    if (!opened_) return false;
    
    // Convert to target format if needed
    std::vector<uint8_t*> output_planes(format_.channels);
    if (!convert_from_float(planes, nb_samples, output_planes)) {
        return false;
    }
    
    // Write to FIFO
    int ret = av_audio_fifo_write(fifo_, reinterpret_cast<void**>(output_planes.data()), 
                                  nb_samples);
    if (ret < 0) {
        return false;
    }
    
    // Encode and write when we have enough samples
    int frame_size = codec_ctx_->frame_size;
    while (av_audio_fifo_size(fifo_) >= frame_size) {
        AVFrame* frame = av_frame_alloc();
        if (!frame) return false;
        
        frame->nb_samples = frame_size;
        av_channel_layout_default(&frame->ch_layout, format_.channels);
        frame->sample_rate = format_.sample_rate;
        frame->format = format_.sample_format;
        frame->pts = pts_;
        pts_ += frame_size;
        
        ret = av_frame_get_buffer(frame, 0);
        if (ret < 0) {
            av_frame_free(&frame);
            return false;
        }
        
        ret = av_audio_fifo_read(fifo_, reinterpret_cast<void**>(frame->data), frame_size);
        if (ret < 0) {
            av_frame_free(&frame);
            return false;
        }
        
        bool success = encode_and_write(frame);
        av_frame_free(&frame);
        
        if (!success) return false;
    }
    
    return true;
}

bool AudioWriter::write_frame_async(std::vector<float*>&& planes, int nb_samples) {
    if (!opened_) return false;
    return frame_queue_->push(std::move(planes), nb_samples);
}

bool AudioWriter::convert_from_float(const std::vector<float*>& input, int nb_samples,
                                    std::vector<uint8_t*>& output) {
    if (format_.sample_format == AV_SAMPLE_FMT_FLTP) {
        // No conversion needed, just cast
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            output[i] = reinterpret_cast<uint8_t*>(input[i]);
        }
        return true;
    }
    
    // Need to resample/convert
    output.resize(format_.channels);
    int out_samples = swr_get_out_samples(swr_ctx_, nb_samples);
    
    for (int c = 0; c < format_.channels; c++) {
        output[c] = static_cast<uint8_t*>(
            arena_->allocate(out_samples * av_get_bytes_per_sample(format_.sample_format), 64));
    }
    
    int ret = swr_convert(swr_ctx_, output.data(), out_samples,
                         const_cast<const uint8_t**>(reinterpret_cast<uint8_t**>(const_cast<float**>(input.data()))), nb_samples);
    
    return ret > 0;
}

bool AudioWriter::encode_and_write(AVFrame* frame) {
    int ret = avcodec_send_frame(codec_ctx_, frame);
    if (ret < 0) {
        std::cerr << "Error sending frame to encoder" << std::endl;
        return false;
    }
    
    while (true) {
        AVPacket* packet = av_packet_alloc();
        ret = avcodec_receive_packet(codec_ctx_, packet);
        
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_free(&packet);
            break;
        }
        
        if (ret < 0) {
            std::cerr << "Error encoding packet" << std::endl;
            av_packet_free(&packet);
            return false;
        }
        
        // Rescale PTS
        packet->stream_index = stream_index_;
        packet->pts = av_rescale_q(packet->pts, codec_ctx_->time_base, 
                                   format_ctx_->streams[stream_index_]->time_base);
        packet->dts = av_rescale_q(packet->dts, codec_ctx_->time_base,
                                   format_ctx_->streams[stream_index_]->time_base);
        packet->duration = av_rescale_q(packet->duration, codec_ctx_->time_base,
                                        format_ctx_->streams[stream_index_]->time_base);
        
        ret = av_interleaved_write_frame(format_ctx_, packet);
        av_packet_free(&packet);
        
        if (ret < 0) {
            std::cerr << "Error writing packet" << std::endl;
            return false;
        }
        
        frames_written_++;
        samples_written_ += frame->nb_samples;
    }
    
    return true;
}

void AudioWriter::encode_thread() {
    std::vector<float*> planes;
    int nb_samples;
    
    while (frame_queue_->pop(planes, nb_samples, 100) || !stop_encode_) {
        if (!planes.empty()) {
            write_frame(planes, nb_samples);
            planes.clear();
        }
    }
}

void AudioWriter::close() {
    if (!opened_) return;
    
    // Signal encoder thread to stop
    frame_queue_->set_eof();
    stop_encode_ = true;
    
    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
    
    // Flush FIFO
    if (fifo_ && av_audio_fifo_size(fifo_) > 0) {
        int frame_size = codec_ctx_ ? codec_ctx_->frame_size : 1024;
        while (av_audio_fifo_size(fifo_) > 0) {
            AVFrame* frame = av_frame_alloc();
            if (!frame) break;
            
            int samples_to_read = std::min(av_audio_fifo_size(fifo_), frame_size);
            frame->nb_samples = samples_to_read;
            av_channel_layout_default(&frame->ch_layout, format_.channels);
            frame->sample_rate = format_.sample_rate;
            frame->format = format_.sample_format;
            frame->pts = pts_;
            pts_ += samples_to_read;
            
            if (av_frame_get_buffer(frame, 0) >= 0) {
                av_audio_fifo_read(fifo_, reinterpret_cast<void**>(frame->data), samples_to_read);
                encode_and_write(frame);
            }
            
            av_frame_free(&frame);
        }
    }
    
    // Flush encoder
    if (codec_ctx_) {
        avcodec_send_frame(codec_ctx_, nullptr);
        AVPacket* packet = av_packet_alloc();
        while (avcodec_receive_packet(codec_ctx_, packet) >= 0) {
            packet->stream_index = stream_index_;
            av_interleaved_write_frame(format_ctx_, packet);
            av_packet_unref(packet);
        }
        av_packet_free(&packet);
    }
    
    // Write trailer
    if (format_ctx_) {
        av_write_trailer(format_ctx_);
    }
    
    // Clean up
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
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    if (format_ctx_) {
        if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx_->pb);
        }
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
    }
    
    opened_ = false;
    encoder_opened_ = false;
    hw_accel_enabled_ = false;
}

} // namespace nass_x
