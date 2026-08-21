#include <iostream>
#include <chrono>
#include <random>
#include "nass/core/types.hpp"
#include "nass/math/stft.hpp"
#include "nass/memory/arena.hpp"

using namespace nass;

int main() {
    std::cout << "=== NASS-X STFT Benchmark ===" << std::endl << std::endl;
    
    // Configuration - use smaller size for quick testing
    STFTConfig config;
    config.fft_size = 512;
    config.hop_size = 128;
    config.win_size = 512;
    
    const size_t num_samples = 44100 * 1;  // 1 second at 44.1kHz
    
    // Generate random audio
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<> dist(-1.0f, 1.0f);
    
    FloatBuffer audio(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        audio[i] = dist(gen);
    }
    
    // Initialize STFT engine
    STFT stft(config);
    if (!stft.is_valid()) {
        std::cout << "ERROR: STFT initialization failed" << std::endl;
        return 1;
    }
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  FFT Size: " << config.fft_size << std::endl;
    std::cout << "  Hop Size: " << config.hop_size << std::endl;
    std::cout << "  Audio Duration: " << (num_samples / 44100.0) << " seconds" << std::endl;
    std::cout << std::endl;
    
    // Warm-up run
    auto warmup_result = stft.transform(audio.data(), num_samples);
    if (!warmup_result.ok()) {
        std::cout << "ERROR: Warm-up transform failed" << std::endl;
        return 1;
    }
    
    // Benchmark with fewer runs
    const int num_runs = 3;
    double total_time_ms = 0.0;
    
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = stft.transform(audio.data(), num_samples);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;
        
        total_time_ms += time_ms;
        
        std::cout << "Run " << (i + 1) << ": " << time_ms << " ms" << std::endl;
        
        if (!result.ok()) {
            std::cout << "ERROR: Transform failed on run " << (i + 1) << std::endl;
            return 1;
        }
    }
    
    double avg_time_ms = total_time_ms / num_runs;
    double audio_duration_sec = num_samples / 44100.0;
    double processing_speed = audio_duration_sec / (avg_time_ms / 1000.0);
    
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Average Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Real-time Factor: " << processing_speed << "x" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Note: This is using naive DFT implementation." << std::endl;
    std::cout << "Performance will improve significantly with KissFFT/MKL backend." << std::endl;
    
    return 0;
}
