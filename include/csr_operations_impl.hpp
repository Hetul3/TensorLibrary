#ifndef CSR_OPERATIONS_IMPL_HPP
#define CSR_OPERATIONS_IMPL_HPP

#include "csr_operations.hpp"
#include <unordered_map>

// convert CSR object to an xarray
template <typename T>
xt::xarray<T> CSRToDense(const CSR<T> &csr)
{
    xt::xarray<T> tensor = xt::zeros<T>(csr.getShape());

    for (size_t i = 0; i < csr.getValues().size(); ++i)
    {
        tensor[csr.getIndices()[i]] = csr.getValues()[i];
    }

    return tensor;
}

// convert xarray to CSR object
template <typename T>
CSR<T> DenseToCSR(const xt::xarray<T> &tensor)
{
    CSR<T> csr(tensor);
    return csr;
}

// multiply two CSR objects
template <typename T>
CSR<T> CSRMult(const CSR<T> &csr1, const CSR<T> &csr2)
{
    if (!_areMultiplicable(csr1, csr2))
    {
        throw std::invalid_argument("CSR objects are not compatible for multiplication");
    }

    // Resulting shape
    std::vector<size_t> resultShape(csr1.shape);
    resultShape.back() = csr2.shape.back();

    // result storage
    std::vector<double> resultValues;
    std::vector<std::vector<size_t>> resultIndices(resultShape.size());

    // Map csr2 indices for quick lookup
    std::unordered_map<size_t, std::vector<size_t>> indexMapB;
    for (size_t i = 0; i < csr2.values.size(); ++i)
    {
        size_t key = csr2.indices.front()[i];
        indexMapB[key].push_back(i);
    }

    // Multiply non zero values
    for (szie_t i = 0; i < csr1.values.size(); ++i)
    {
        size_t key = csr1.indices.back()[i];

        // Find matching indices
        auto it = indexMapB.find(key);
        if (it != indexMapB.end())
        {
            for (size_t j : it->second)
            {
                // Combine indices with broadcasting logic
                std::vector<size_t> resultIndex(resultShape.size());
                for (size_t dim = 0; dim < resultShape.size(); ++dim)
                {
                    size_t indexA = dim < csr1.indices.size() ? csr1.indices[dim][i] : 0;
                    size_t indexB = dim < csr2.indices.size() ? csr2.indices[dim][j] : 0;

                    // Apply broadcasting
                    resultIndex[dim] = (csr1.shape[dim] == 1) ? indexB : (csr2.shape[dim] == 1) ? indexA
                                                                                                : indexA;
                }

                // Compute the value and store it
                T value = csr1.values[i] * csr2.values[j];
                resultValues.push_back(value);
                for (size_t dim = 0; dim < resultShape.size(); ++dim)
                {
                    resultIndices[dim].push_back(resultIndex[dim]);
                }
            }
        }
    }

    return CSR<T>{resultShape, resultValues, resultIndices};
}

// Anonymous namespace
namespace
{
    template <typename T>
    bool _areMultiplicable(const CSR<T> &csr1, const CSR<T> &csr2)
    {
        if (csr1.shape.back() != csr2.shape.front())
        {
            return false;
        }

        // check for broadcasting
        size_t dimA = csr1.shape.size();
        size_t dimB = csr2.shape.size();
        size_t maxDims = std::max(dimA, dimB);

        for (size_t i = 1; i < maxDims; ++i)
        {
            size_t dimA_index = (i < dimA) ? csr1.shape[dimA - 1 - i] : 1;
            size_t dimB_index = (i < dimB) ? csr2.shape[dimB - 1 - i] : 1;

            if (dimA_index != 1 && dimB_index != 1 && dimA_index != dimB_index)
            {
                return false;
            }
        }
        return true;
    }

}

#endif // CSR_OPERATIONS_IMPL_HPP