#include "xtensor_operations.hpp"

namespace sparse_ops {
    // Explicit template instantiations for common types
    template bool sparse_ops::isSparse<xt::xarray<double>>(const xt::xarray<double>&, double);
    template bool sparse_ops::isSparse<xt::xtensor<float, 2>>(const xt::xtensor<float, 2>&, double);

    template double sparse_ops::sparsity<xt::xarray<double>>(const xt::xarray<double>&);
    template double sparse_ops::sparsity<xt::xtensor<float, 2>>(const xt::xtensor<float, 2>&);
}
