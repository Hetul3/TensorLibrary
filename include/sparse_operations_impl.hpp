#ifndef SPARSE_OPERATIONS_IMPL_HPP
#define SPARSE_OPERATIONS_IMPL_HPP

#include "sparse_operations.hpp"
#include <xtensor/xarray.hpp>
#include <vector>
#include <stdexcept>
#include <tuple>

namespace sparse_ops
{
    template <typename Tensor>
    bool isSparse(const Tensor &tensor, double threshold)
    {
        /*
        Two approaches:
        1. xt::equal and xt::sum are highly optimized in xtensor making it faster than a linear scan,
           however, it creates a temporary tensor which can be memory intensive
        2. Linear scan with early stopping doesn't create a temporary tensor but is slower
        */
        // Approach 1
        auto zero_count = xt::sum(xt::equal(tensor, 0.0));
        double sparsity = static_cast<double>(zero_count()) / tensor.size();
        return sparsity >= threshold;
        // Approach 2
        /*
        size_t zero_count = 0;
        size_t max_zeros = static_cast<size_t>(tensor.size() * threshold);
          for (const auto& val : tensor) {
             if (val == 0.0) {
                 zero_count++;
                 if(zero_count >= max_zeros) {
                     return true;
                 }
             }
          }
          return false;
        */
    }
    template <typename Tensor>
    double sparsity(const Tensor &tensor)
    {
        auto zero_count = xt::sum(xt::equal(tensor, 0.0));
        return static_cast<double>(zero_count()) / tensor.size();
    }

    // Private helper to convert to xarray to CSR format, generalized for any tensor shape
    template <typename Tensor>
    auto _toCompressedFormat(const Tensor &tensor) -> std::tuple<std::vector<double>, std::vector<std::vector<size_t>>>
    {
        // storing all non zero values
        // linear list of all non zero values in the tensor
        std::vector<double> values;
        // indices of the non zero values for each dimension of the tensor
        /*
        Size of indices is # of dimensions * # of non zero values.
        The nth vector shows the index of all the non zero values in the nth dimension.
        */
        std::vector<std::vector<size_t>> indices(tensor.dimension());

        auto shape = tensor.shape();
        size_t = total_elements = tensor.size();
        for (size_t flat_index = 0; flat_index < total_elements; ++flat_index)
        {
            // Map flat index to dimensional index, ex. index 3 in (2,3) tensor is (1,0)
            std::vector<size_t> multi_index(tensor.dimension());
            size_t temp = flat_index;
            // figure out the multi index from the flat index
            for (size_t dim = tensor.dimension(); dim > 0; --dim)
            {
                size_t current_dim = dim - 1;
                multi_index[current_dim] = temp % shape[current_dim];
                temp /= shape[current_dim];
            }

            // Check if the value is non zero
            if (tensor(flat_index) != 0.0)
            {
                values.push_back(tensor(flat_index));
                for (size_t dim = 0; dim < tensor.dimension(); ++dim)
                {
                    indices[dim].push_back(multi_index[dim]);
                }
            }
        }
        return {values, indices};
    }

    // check if the dimensions of the tensors are compatible for multiplication
    template <typename T>
    bool areTensorsMultiplicable(const xt::xarray<T> tensorA, const xt::xarray<T> tensorB)
    {
        // get shapes of the tensors
        std::vector<size_t> shapeA = tensorA.shape();
        std::vector<size_t> shapeB = tensorB.shape();

        // Ensure valid dimensions
        if (shapeA.empty() || shapeB.empty())
        {
            return false;
        }

        // Check if the last dimension of tensorA is equal to the first dimension of tensorB
        if (shapeA[shapeA.size() - 1] != shapeB[0])
        {
            return false;
        }

        // Check for broadcasting
        size_t dimA = shapeA.size();
        size_t dimB = shapeB.size();
        size_t maxDims = std::max(dimA, dimB);
        for (size_t i = 1; i <= maxDims - 1; ++i)
        {
            size_t dimA_index = (i <= dimA - 1) ? shapeA[dimA - 1 - i] : 1;
            size_t dimB_index = (i <= dimB - 1) ? shapeB[dimB - 1 - i] : 1;

            if (dimA_index != dimB_index)
            {
                return false;
            }
        }

        return true;
    }

    auto _multiplyCompressedFormat(const xt::xarray<double> tensorA, const xt::xarray<double> tensorB) -> xt::xarray<double>
    {
        // convert tensorA and tensorB to CSR format
        auto [valuesA, indicesA] = _toCompressedFormat(tensorA);
        auto [valuesB, indicesB] = _toCompressedFormat(tensorB);

        // check dimension compatibility
        if(!areTensorsMultiplicable(tensorA, tensorB)) {
            throw std::invalid_argument("Tensors are not compatible for multiplication");
        }   
    }
}

#endif // SPARSE_OPERATIONS_IMPL_HPP