#ifndef SPARSE_OPERATIONS_HPP
#define SPARSE_OPERATIONS_HPP

#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

namespace sparse_ops {
    // operations to check if a tensor/array is sparse based on threshold, and operation to return sparsity percentage
    template <typename Tensor>
    bool isSparse(const Tensor& tensor, double threshold=0.8);

    template <typename Tensor>
    double sparsity(const Tensor& tensor);

    auto multiplyCompressedFormat(const xt::xarray<double>& tensorA, const xt::xarray<double>& tensorB) -> xt::xarray<double>;
}

namespace {
    // helper functions
    template <typename Tensor>
    auto _toCompressedFormat(const Tensor& tensor) -> std::tuple<std::vector<double>, std::vector<std::vector<size_t>>>;

    bool _areTensorsMultiplicable(const xt::xarray<double>& tensorA, const xt::xarray<double>& tensorB);

    bool _worthUsingSparse(const xt::xarray<double>& tensorA, const xt::xarray<double>& tensorB);
}

#include "sparse_operations_impl.hpp"

#endif // SPARSE_OPERATIONS_HPP