#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include "src/core/wav_utils.hpp"

// Constants for STFT
const int FFT_SIZE = 2048;
const int HOP_SIZE = 512; // 75% overlap
const int WINDOW_SIZE = FFT_SIZE;

// Generate Hann window
std::vector<float> create_hann_window(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    return window;
}

// STFT Analysis
std::vector<std::vector<std::complex<float>>> stft(const std::vector<float>& input, 
                                                    const std::vector<float>& window) {
    int num_frames = (input.size() - WINDOW_SIZE) / HOP_SIZE + 1;
    std::vector<std::vector<std::complex<float>>> spectrum(num_frames, 
        std::vector<std::complex<float>>(FFT_SIZE / 2 + 1));

    // Setup FFTW plan
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (FFT_SIZE / 2 + 1));
    float* in = (float*)fftwf_malloc(sizeof(float) * FFT_SIZE);
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(FFT_SIZE, in, out, FFTW_ESTIMATE);

    for (int frame = 0; frame < num_frames; ++frame) {
        int start = frame * HOP_SIZE;
        
        // Apply window
        for (int i = 0; i < FFT_SIZE; ++i) {
            in[i] = input[start + i] * window[i];
        }

        // Execute FFT
        fftwf_execute(plan);

        // Store result (normalized by FFT_SIZE as float)
        float norm = 1.0f / static_cast<float>(FFT_SIZE);
        for (int i = 0; i <= FFT_SIZE / 2; ++i) {
            spectrum[frame][i] = std::complex<float>(out[i][0] * norm, out[i][1] * norm);
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    return spectrum;
}

// ISTFT Synthesis with Overlap-Add
std::vector<float> istft(const std::vector<std::vector<std::complex<float>>>& spectrum,
                         const std::vector<float>& window, size_t expected_size) {
    int num_frames = spectrum.size();
    int output_size = (num_frames - 1) * HOP_SIZE + WINDOW_SIZE;
    
    // Allocate output with extra padding for safety
    std::vector<float> output(output_size, 0.0f);
    std::vector<float> window_sum(output_size, 0.0f);

    // Setup FFTW plan
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (FFT_SIZE / 2 + 1));
    float* out = (float*)fftwf_malloc(sizeof(float) * FFT_SIZE);
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(FFT_SIZE, in, out, FFTW_ESTIMATE);

    for (int frame = 0; frame < num_frames; ++frame) {
        int start = frame * HOP_SIZE;

        // Prepare input for IFFT (conjugate symmetric for real output)
        in[0][0] = spectrum[frame][0].real(); in[0][1] = 0.0f; // DC
        for (int i = 1; i < FFT_SIZE / 2; ++i) {
            in[i][0] = spectrum[frame][i].real();
            in[i][1] = spectrum[frame][i].imag();
        }
        in[FFT_SIZE/2][0] = spectrum[frame][FFT_SIZE/2].real(); 
        in[FFT_SIZE/2][1] = 0.0f; // Nyquist

        // Execute IFFT (FFTW scales by N, so we need to normalize)
        fftwf_execute(plan);

        // Apply window and overlap-add
        // FFTW c2r scales by N, so out[i] is already N times larger
        // We already normalized in STFT by 1/N, so here we just apply window
        for (int i = 0; i < FFT_SIZE && (start + i) < output_size; ++i) {
            output[start + i] += out[i] * window[i];
            window_sum[start + i] += window[i] * window[i];
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    // Normalize by window sum (avoid division by zero)
    for (size_t i = 0; i < output_size; ++i) {
        if (window_sum[i] > 1e-8f) {
            output[i] /= window_sum[i];
        } else {
            output[i] = 0.0f;
        }
    }

    // Resize to expected size (trim or pad)
    output.resize(expected_size, 0.0f);
    return output;
}

int main() {
    std::cout << "=== PHASE 1 VERIFICATION: PERFECT RECONSTRUCTION ===" << std::endl;

    // 1. Generate Test Signal (Sine Wave) - use exact multiple of HOP_SIZE for clean frames
    const float duration = 2.0f;
    const uint32_t sample_rate = 44100;
    const float freq = 440.0f;
    
    // Ensure signal length is compatible with STFT framing
    size_t base_samples = static_cast<size_t>(duration * sample_rate);
    size_t num_frames_needed = (base_samples - WINDOW_SIZE) / HOP_SIZE + 1;
    size_t num_samples = (num_frames_needed - 1) * HOP_SIZE + WINDOW_SIZE;
    
    std::vector<float> input_signal(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        input_signal[i] = 0.5f * std::sin(2.0f * M_PI * freq * i / sample_rate);
    }

    // 2. Save to WAV
    std::string wav_path = "test_input.wav";
    if (!nass_x::WavUtils::write(wav_path, input_signal, sample_rate)) {
        std::cerr << "Failed to write WAV file!" << std::endl;
        return 1;
    }
    std::cout << "Generated test signal: " << wav_path << " (" << num_samples << " samples)" << std::endl;

    // 3. Read back from WAV (Verify I/O)
    std::vector<float> loaded_signal;
    uint32_t loaded_sr;
    if (!nass_x::WavUtils::read(wav_path, loaded_signal, loaded_sr)) {
        std::cerr << "Failed to read WAV file!" << std::endl;
        return 1;
    }
    std::cout << "Loaded signal from WAV: " << loaded_signal.size() << " samples @ " << loaded_sr << "Hz" << std::endl;

    // Verify I/O roundtrip
    float io_error = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        float diff = std::abs(input_signal[i] - loaded_signal[i]);
        if (diff > io_error) io_error = diff;
    }
    std::cout << "I/O Roundtrip Max Error: " << io_error << std::endl;
    if (io_error > 1e-6f) {
        std::cerr << "FAIL: I/O roundtrip error too high!" << std::endl;
        return 1;
    }

    // 4. Create Window
    std::vector<float> window = create_hann_window(WINDOW_SIZE);

    // 5. STFT Analysis
    std::cout << "Running STFT..." << std::endl;
    auto spectrum = stft(loaded_signal, window);
    int num_frames = spectrum.size();
    std::cout << "STFT Complete: " << num_frames << " frames" << std::endl;

    // 6. ISTFT Synthesis (Identity Processing)
    std::cout << "Running ISTFT..." << std::endl;
    auto reconstructed = istft(spectrum, window, num_samples);
    std::cout << "ISTFT Complete: " << reconstructed.size() << " samples" << std::endl;

    // 7. Compare Input vs Reconstructed (skip edge regions where window sum is incomplete)
    // For Hann window with 75% overlap, edges should be fine after wsum normalization
    size_t skip_start = WINDOW_SIZE;  // Skip first window where wsum ramps up
    size_t skip_end = WINDOW_SIZE;    // Skip last window where wsum ramps down
    size_t compare_start = skip_start;
    size_t compare_end = std::min(num_samples - skip_end, reconstructed.size());
    
    float max_error = 0.0f;
    float mse = 0.0f;
    size_t count = 0;
    
    for (size_t i = compare_start; i < compare_end; ++i) {
        float diff = std::abs(loaded_signal[i] - reconstructed[i]);
        if (diff > max_error) max_error = diff;
        mse += diff * diff;
        count++;
    }
    
    if (count > 0) {
        mse = std::sqrt(mse / count);
    }

    std::cout << "\n=== RECONSTRUCTION METRICS ===" << std::endl;
    std::cout << "Comparison range: [" << compare_start << ", " << compare_end << ")" << std::endl;
    std::cout << "Max Absolute Error (L-inf): " << max_error << std::endl;
    std::cout << "RMSE: " << mse << std::endl;
    std::cout << "Threshold: 1e-5" << std::endl;

    if (max_error < 1e-5f) {
        std::cout << "\n✅ PHASE 1 PASSED: Perfect Reconstruction Achieved!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ PHASE 1 FAILED: Reconstruction error exceeds threshold." << std::endl;
        // Dump some samples for debugging
        std::cout << "Debug (samples 1000-1005):" << std::endl;
        for (size_t i = 1000; i < 1005 && i < num_samples; ++i) {
            std::cout << "  In: " << loaded_signal[i] << " Rec: " << reconstructed[i] 
                      << " Diff: " << std::abs(loaded_signal[i] - reconstructed[i]) << std::endl;
        }
        return 1;
    }
}
