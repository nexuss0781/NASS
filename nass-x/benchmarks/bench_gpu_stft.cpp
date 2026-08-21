#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <complex>
#include "nass_x/gpu/cuda_backend.hpp"
#include "nass_x/gpu/fft_cuda.hpp"
#include "nass_x/gpu/gpu_engine.hpp"

using namespace nass_x::gpu;

void print_separator() {
    std::cout << "============================================================" << std::endl;
}

void benchmark_gpu_stft() {
    print_separator();
    std::cout << "NASS-X Phase 3: GPU Acceleration Benchmark" << std::endl;
    print_separator();
    
    // Check GPU availability
    if (!CudaBackend::is_available()) {
        std::cout << "\nNo CUDA devices found. Skipping GPU benchmarks." << std::endl;
        std::cout << "To run GPU benchmarks, ensure NVIDIA GPU and CUDA toolkit are installed." << std::endl;
        return;
    }
    
    std::cout << "\nCUDA Devices Available: " << CudaBackend::get_device_count() << std::endl;
    
    // Initialize GPU engine
    try {
        GpuEngine engine(0);
        
        if (!engine.is_ready()) {
            std::cout << "\nGPU engine failed to initialize." << std::endl;
            return;
        }
        
        // Benchmark parameters
        std::vector<int> fft_sizes = {512, 1024, 2048, 4096};
        std::vector<int> batch_sizes = {1, 8, 32, 128, 512};
        int hop_size = 256; // Default hop size
        
        std::cout << "\nRunning benchmarks..." << std::endl;
        print_separator();
        
        for (int fft_size : fft_sizes) {
            std::cout << "\nFFT Size: " << fft_size << std::endl;
            std::cout << "------------------------------------------------------------" << std::endl;
            
            for (int batch_size : batch_sizes) {
                // Generate random audio data
                size_t samples_per_frame = fft_size;
                size_t total_samples = batch_size * samples_per_frame;
                
                std::vector<float> audio(total_samples);
                std::vector<std::complex<float>> spectrogram(batch_size * (fft_size / 2 + 1));
                
                // Fill with random data
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
                
                for (size_t i = 0; i < total_samples; ++i) {
                    audio[i] = dist(gen);
                }
                
                // Warm-up run
                engine.execute_stft(audio.data(), spectrogram.data(), batch_size, fft_size, hop_size);
                
                // Benchmark runs
                const int num_runs = 10;
                double total_time = 0.0;
                
                for (int run = 0; run < num_runs; ++run) {
                    double elapsed = engine.execute_stft(audio.data(), spectrogram.data(), 
                                                        batch_size, fft_size, hop_size);
                    total_time += elapsed;
                }
                
                double avg_time = total_time / num_runs;
                double throughput = (batch_size * fft_size) / (avg_time * 1e-3); // samples/sec
                double realtime_factor = throughput / 48000.0; // Assuming 48kHz audio
                
                std::printf("  Batch=%4d | Avg Latency=%7.3f ms | Throughput=%10.0f samples/s | %.1fx Real-time\n",
                           batch_size, avg_time, throughput, realtime_factor);
            }
        }
        
        // Print statistics
        print_separator();
        auto stats = engine.get_stats();
        std::cout << "\nEngine Statistics:" << std::endl;
        std::cout << "  Total Operations: " << stats.total_operations << std::endl;
        std::cout << "  Average Latency: " << stats.avg_latency_ms << " ms" << std::endl;
        std::cout << "  Peak Throughput: " << stats.peak_throughput_gflops << " GFLOPS" << std::endl;
        std::cout << "  Data Transferred: " << (stats.bytes_transferred / (1024 * 1024)) << " MB" << std::endl;
        
        print_separator();
        std::cout << "\nBenchmark Complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nBenchmark failed: " << e.what() << std::endl;
    }
}

void test_istft_roundtrip() {
    print_separator();
    std::cout << "Testing STFT/iSTFT Round-trip Reconstruction" << std::endl;
    print_separator();
    
    if (!CudaBackend::is_available()) {
        std::cout << "No CUDA devices found. Skipping reconstruction test." << std::endl;
        return;
    }
    
    try {
        GpuEngine engine(0);
        
        if (!engine.is_ready()) {
            std::cout << "GPU engine not ready." << std::endl;
            return;
        }
        
        int fft_size = 1024;
        int batch_size = 8;
        int hop_size = 256;
        
        // Generate test signal
        size_t samples = batch_size * fft_size;
        std::vector<float> original(samples);
        std::vector<std::complex<float>> spectrogram(batch_size * (fft_size / 2 + 1));
        std::vector<float> reconstructed(samples);
        
        // Create sine wave test signal
        float frequency = 440.0f; // A4 note
        float sample_rate = 48000.0f;
        
        for (size_t i = 0; i < samples; ++i) {
            float t = i / sample_rate;
            original[i] = 0.5f * std::sin(2.0f * 3.14159f * frequency * t);
        }
        
        // STFT
        double stft_time = engine.execute_stft(original.data(), spectrogram.data(), 
                                               batch_size, fft_size, hop_size);
        std::cout << "STFT completed in " << stft_time << " ms" << std::endl;
        
        // iSTFT
        double istft_time = engine.execute_istft(spectrogram.data(), reconstructed.data(),
                                                 batch_size, fft_size, hop_size);
        std::cout << "iSTFT completed in " << istft_time << " ms" << std::endl;
        
        // Calculate reconstruction error (simplified - doesn't account for overlap-add)
        double mse = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            double diff = original[i] - reconstructed[i];
            mse += diff * diff;
        }
        mse /= samples;
        
        std::cout << "Mean Squared Error: " << mse << std::endl;
        std::cout << "Note: Perfect reconstruction requires proper overlap-add implementation" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Reconstruction test failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    
    bool run_benchmark = true;
    bool run_test = true;
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--benchmark" || arg == "-b") {
            run_test = false;
        } else if (arg == "--test" || arg == "-t") {
            run_benchmark = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -b, --benchmark   Run only benchmarks" << std::endl;
            std::cout << "  -t, --test        Run only reconstruction test" << std::endl;
            std::cout << "  -h, --help        Show this help message" << std::endl;
            return 0;
        }
    }
    
    if (run_benchmark) {
        benchmark_gpu_stft();
    }
    
    if (run_test) {
        std::cout << "\n";
        test_istft_roundtrip();
    }
    
    std::cout << "\n";
    print_separator();
    
    return 0;
}
