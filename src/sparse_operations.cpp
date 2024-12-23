#include "sparse_operations.hpp"

namespace sparse_ops {
    xt::xarray<double> addTensor(const xt::xarray<double>& tensor1, const xt::xarray<double>& tensor2) {
        if(tensor1.shape() != tensor2.shape()) {
            throw std::invalid_argument("Tensors must have the same shape for addition operation");
        }
        return tensor1 + tensor2;
    }
}