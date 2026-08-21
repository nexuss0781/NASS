// Tensor implementation - mostly header-only
#include "nass/tensor/tensor.hpp"

namespace nass {

// Explicit template instantiations
template class Tensor<Float32, 4>;
template class Tensor<Float64, 4>;
template class Tensor<ComplexFloat, 4>;
template class Tensor<Int16, 4>;

} // namespace nass
