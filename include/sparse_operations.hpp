#ifndef SPARSE_OPERATIONS_HPP
#define SPARSE_OPERATIONS_HPP

#include <xtensor/xarray.hpp>
namespace sparse_ops {
    // Function for densor tensor addition as an initial test
    xt::xarray<double> addTensor(const xt::xarray<double>& tensor1, const xt::xarray<double>& tensor2);
}

#endif // SPARSE_OPERATIONS_HPP