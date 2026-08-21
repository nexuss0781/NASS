#ifndef NASS_CORE_CONFIG_HPP
#define NASS_CORE_CONFIG_HPP

#include <string>
#include <optional>
#include <thread>
#include "nass/core/types.hpp"

namespace nass {

// Global configuration for NASS-X engine
struct EngineConfig {
    // Processing parameters
    size_t num_threads = 0;  // 0 = auto-detect CPU cores
    size_t chunk_size_samples = 65536;  // ~1.5s at 44.1kHz
    
    // Memory settings
    bool use_huge_pages = false;
    bool numa_aware = false;
    size_t arena_size_mb = 256;
    
    // Performance tuning
    bool simd_enabled = true;
    int simd_level = 0;  // 0=auto, 1=SSE, 2=AVX2, 3=AVX-512
    
    // Debug options
    bool enable_profiling = false;
    bool verbose_logging = false;
    
    // Default STFT configuration
    STFTConfig default_stft;
    
    EngineConfig() {
        default_stft.fft_size = 2048;
        default_stft.hop_size = 512;
        default_stft.win_size = 2048;
        default_stft.normalized = true;
        default_stft.format = AudioFormat::FLOAT32;
    }
    
    // Auto-detect optimal settings
    static EngineConfig auto_detect() {
        EngineConfig config;
        
        // Detect CPU cores
        unsigned int cores = std::thread::hardware_concurrency();
        config.num_threads = (cores > 0) ? cores : 4;
        
        // Detect SIMD capabilities (runtime detection in implementation)
        config.simd_level = detect_simd_level();
        
        return config;
    }
    
private:
    static int detect_simd_level();
};

// Singleton configuration accessor
class Config {
public:
    static Config& instance() {
        static Config cfg;
        return cfg;
    }
    
    EngineConfig& get() { return config_; }
    const EngineConfig& get() const { return config_; }
    
    void set(const EngineConfig& cfg) { config_ = cfg; }
    
private:
    Config() : config_(EngineConfig::auto_detect()) {}
    EngineConfig config_;
};

} // namespace nass

#endif // NASS_CORE_CONFIG_HPP
