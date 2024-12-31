#ifndef XTENSOR_OPERATIONS_IMPL_HPP
#define XTENSOR_OPERATIONS_IMPL_HPP

#include "xtensor_operations.hpp"
#include <xtensor/xarray.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

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

    // multiplication of two tensors in compressed format
    auto multiplyCompressedFormat(const xt::xarray<double> &tensorA, const xt::xarray<double> &tensorB) -> xt::xarray<double>
    {
        // Convert tensorA and tensorB to CSR format
        auto [valuesA, indicesA] = _toCompressedFormat(tensorA);
        auto [valuesB, indicesB] = _toCompressedFormat(tensorB);

        // Check dimension compatibility
        TensorMultiplicabilityAnalysisStruct analysis = _areTensorsMultiplicable(tensorA, tensorB);
        if (!analysis.isMultiplcable)
        {
            throw std::invalid_argument("Tensors are not compatible for multiplication");
        }

        // Resulting shape
        std::vector<size_t> resultShape(tensorA.shape().begin(), tensorA.shape().end());
        resultShape.back() = tensorB.shape().back();

        // Resulting tensor
        xt::xarray<double> result = xt::zeros<double>(resultShape);

        // Multiply the tensors in compressed format
        size_t numNonZerosValuesA = valuesA.size();
        size_t numNonZerosValuesB = valuesB.size();

        std::unordered_map<size_t, std::vector<size_t>> indexMapB;
        for (size_t j = 0; j < numNonZerosValuesB; ++j)
        {
            indexMapB[indicesB.front()[j]].push_back(j);
        }

        for (size_t i = 0; i < numNonZerosValuesA; ++i)
        {
            // Find matching inner dimensions
            auto it = indexMapB.find(indicesA.back()[i]);
            if (it != indexMapB.end())
            {
                for (size_t j : it->second)
                {
                    if (analysis.requiresBroadcasting)
                    {
                        // create the resulting index with broadcasting logic
                        std::vector<size_t> resultIndex(resultShape.size());
                        for (size_t dim = 0; dim < resultShape.size(); ++dim)
                        {
                            size_t indexA = dim < indicesA.size() ? indicesA[dim][i] : 0;
                            size_t indexB = dim < indicesB.size() ? indicesB[dim][j] : 0;

                            // Apply broadcasting
                            resultIndex[dim] = (tensorA.shape()[dim] == 1) ? indexB : (tensorB.shape()[dim] == 1) ? indexA
                                                                                                                  : indexA;
                        }

                        result(resultIndex) += valuesA[i] * valuesB[j];
                    }
                    else
                    {
                        // No broadcasting logic required, direct indices use and faster operation
                        std::vector<size_t> resultIndex = indicesA[i];
                        resultIndex.back() = indicesB.back()[j];

                        result(resultIndex) += valuesA[i] * valuesB[j];
                    }
                }
            }
        }

        return result;
    }

} // namespace sparse_ops

namespace
{
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
#pragma omp parallel
        {
            std::vector<double> local_values;
            std::vector<std::vector<size_t>> local_indices(tensor.dimension());

#pragma omp for
            for (size_t flat_index = 0; flat_index < total_elements; ++flat_index)
            {
                // Map flat index to dimensional index, ex. index 3 in (2,3) tensor is (1,0)
                std::vector<size_t> multi_index(tensor.dimension());
                size_t temp = flat_index;
                // Figure out the multi index from the flat index
                for (size_t dim = tensor.dimension(); dim > 0; --dim)
                {
                    size_t current_dim = dime - 1;
                    multi_index[current_dim] = temp % shape[current_dim];
                    temp /= shape[current_dim];
                }

                // Check if the value is non zero
                if (tensor(flat_index) != 0.0)
                {
                    local_values.push_back(tensor(flat_index));
                    for (size_t dim = 0; dim < tensor.dimension(); ++dim)
                    {
                        local_indices[dim].push_back(multi_index[dim]);
                    }
                }
            }

#pragma omp critical
            {
                values.insert(values.end(), local_values.begin(), local_values.end());
                for (size_t dim = 0; dim < tensor.dimension(); ++dim)
                {
                    indices[dim].insert(indices[dim].end(), local_indices[dim].begin(), local_indices[dim].end());
                }
            }
        }
        return {values, indices};
    }

    // check if the dimensions of the tensors are compatible for multiplication
    TensorMultiplicabilityAnalysisStruct _areTensorsMultiplicable(const xt::xarray<double> &tensorA, const xt::xarray<double> &tensorB)
    {
        // get shapes of the tensors
        std::vector<size_t> shapeA(tensorA.shape().begin(), tensorA.shape().end());
        std::vector<size_t> shapeB(tensorB.shape().begin(), tensorB.shape().end());

        TensorMultiplicabilityAnalysisStruct result = {true, false};

        // Ensure valid dimensions
        if (shapeA.empty() || shapeB.empty())
        {
            result.isMultiplcable = false;
            return result;
        }

        // Check if the last dimension of tensorA is equal to the first dimension of tensorB
        if (shapeA.back() != shapeB.front())
        {
            result.isMultiplcable = false;
            return result;
        }

        // Check for broadcasting
        size_t dimA = shapeA.size();
        size_t dimB = shapeB.size();
        size_t maxDims = std::max(dimA, dimB);

        for (size_t i = 1; i <= maxDims - 1; ++i)
        {
            size_t dimA_index = (i <= dimA - 1) ? shapeA[dimA - 1 - i] : 1;
            size_t dimB_index = (i <= dimB - 1) ? shapeB[dimB - 1 - i] : 1;

            if (dimA_index != dimB_index && dimA_index != 1 && dimB_index != 1)
            {
                result.isMultiplcable = false;
                return result;
            }

            if (dimA_index == 1 || dimB_index == 1)
            {
                result.requiresBroadcasting = true;
            }
        }

        return result;
    }

    /*
    Without hardware and processing optimizations (will be considered and implemented later):
    xarray multiplication for dense tensors follows a normal routine for matrix multiplication.
    For higher dimensional tensors, it performs tensor constraction algorithms to convert the tensor
    into matrices and perform matrix multiplication before reshaping to the higher dimensional result.

    For a matrix multiplication of A (MxK) and B (KxN), the time complexity is O(M*K*N).
    For a higher dimensional tensor, with tensor contraction, the runtime is based on size of the resulting tensor,
    O(size of resulting tensor + reshaping overhead)

    For sparse tensor implementation, the runtime is based on converting to CSR format and multiplication of non zero values.
    Runtime becomes O(size of A + size of B + nnz(A) * nnz(B)), making it faster for cases where the tensors are very sparse and
    the size of the resulting tensor is much greater than the size of the input tensors. nnz << size of tensor, say 0.05 * size of the tensor.
    */

    bool _worthUsingSparse(const xt::xarray<double> &tensorA, const xt::xarray<double> &tensorB)
    {
        // calculating terms of sparsity runtime calculation
        double sparsityA = sparse_ops::sparsity(tensorA);
        double sparsityB = sparse_ops::sparsity(tensorB);

        size_t total_elementsA = tensorA.size();
        size_t nnzA = static_cast<size_t>(total_elementsA * sparsityA);

        size_t total_elementsB = tensorB.size();
        size_t nnzB = static_cast<size_t>(total_elementsB * sparsityB);

        size_t sparse_runtime = total_elementsA + total_elementsB + nnzA * nnzB;

        // calculating terms for the dense runtime calculation

        // finding resulting tensor size
        auto shapeA = tensorA.shape();
        auto shapeB = tensorB.shape();

        std::vector<size_t> resultShape;
        resultShape.insert(resultShape.end(), shapeA.begin(), shapeA.end() - 1);
        resultShape.insert(resultShape.end(), shapeB.begin() + 1, shapeB.end());

        size_t resultSize = 1;
        for (const auto &dim : resultShape)
        {
            resultSize *= dim;
        }

        return sparse_runtime < resultSize;
    }
}

#endif // XTENSOR_OPERATIONS_IMPL_HPP