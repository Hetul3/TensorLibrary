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
}

#endif // SPARSE_OPERATIONS_HPP