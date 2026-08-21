#include "nass/core/config.hpp"
#include <thread>

namespace nass {

int EngineConfig::detect_simd_level() {
#if defined(__AVX512F__)
    return 3;  // AVX-512
#elif defined(__AVX2__)
    return 2;  // AVX2
#elif defined(__SSE2__)
    return 1;  // SSE
#else
    return 0;  // None
#endif
}

} // namespace nass
