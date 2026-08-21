// Tensor implementation - mostly header-only
#include "nass_x/tensor/tensor.hpp"

namespace nass_x::tensor {

// Explicit template instantiations
template class Tensor<Float32, 4>;
template class Tensor<Float64, 4>;
template class Tensor<ComplexFloat, 4>;
template class Tensor<Int16, 4>;

} // namespace nass_x::tensor
