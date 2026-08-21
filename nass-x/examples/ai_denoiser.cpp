/**
 * @brief Example: AI-powered audio denoising using NASS-X Neural Engine
 */

#include "nass_x/ai/neural_engine.hpp"
#include "nass_x/ai/audio_processor.hpp"
#include "nass_x/io/audio_reader.hpp"
#include "nass_x/io/audio_writer.hpp"
#include "nass_x/tensor/tensor.hpp"
#include <iostream>
#include <string>

using namespace nass_x;
using namespace nass_x::ai;

int main(int argc, char* argv[]) {
    std::cout << "=== NASS-X AI Audio Denoiser Demo ===" << std::endl;
    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.wav> <output.wav>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    
    ModelManager model_mgr;
    
    std::cout << "[1/5] Loading neural model: " << model_path << std::endl;
#ifdef NASS_X_USE_ONNX
    if (!model_mgr.load_model("denoiser", model_path, true)) {
        std::cerr << "Failed to load model." << std::endl;
        return 1;
    }
#else
    std::cerr << "ONNX Runtime not compiled. Rebuild with -DUSE_ONNX=ON" << std::endl;
    return 1;
#endif
    
    AudioPreprocessor::Config config;
    config.sample_rate = 48000.0f;
    config.channels = 1;
    config.normalize = true;
    
    AudioPreprocessor processor(config);
    
    std::cout << "[2/5] Opening audio files..." << std::endl;
    AudioReader reader(input_path);
    if (!reader.open()) {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }
    
    AudioWriter writer(output_path);
    AudioSpec spec = reader.get_spec();
    if (!writer.open(spec)) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }
    
    auto* engine = model_mgr.get_engine("denoiser");
    if (!engine) {
        std::cerr << "Engine not found!" << std::endl;
        return 1;
    }
    
    auto input_info = engine->get_input_info();
    std::cout << "[3/5] Model loaded: " << engine->get_backend_name() << std::endl;
    
    const size_t chunk_size = 2048;
    std::vector<float> input_buffer(chunk_size);
    std::vector<float> output_buffer(chunk_size);
    
    size_t total_samples = 0;
    std::cout << "[4/5] Processing audio..." << std::endl;
    
    while (reader.read_samples(input_buffer.data(), chunk_size) > 0) {
        Tensor input_tensor = processor.audio_to_tensor(input_buffer.data(), chunk_size);
        std::vector<int64_t> out_shape = input_tensor.shape();
        Tensor output_tensor(out_shape, DType::FLOAT32);
        
        std::vector<Tensor> inputs = {input_tensor};
        std::vector<Tensor> outputs = {output_tensor};
        
        if (!engine->infer(inputs, outputs)) {
            std::cerr << "Inference failed!" << std::endl;
            return 1;
        }
        
        size_t out_samples = 0;
        processor.tensor_to_audio(outputs[0], output_buffer.data(), out_samples);
        writer.write_samples(output_buffer.data(), out_samples);
        total_samples += out_samples;
    }
    
    std::cout << "[5/5] Complete! Processed " << total_samples << " samples" << std::endl;
    std::cout << "Output: " << output_path << std::endl;
    
    return 0;
}
