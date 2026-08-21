// Buffer implementation - mostly header-only, minimal source needed
#include "nass/memory/buffer.hpp"

namespace nass_x::tensor {

// Explicit template instantiations for common types
template class Buffer<Float32>;
template class Buffer<Float64>;
template class Buffer<Int16>;
template class Buffer<ComplexFloat>;

} // namespace nass_x::tensor
