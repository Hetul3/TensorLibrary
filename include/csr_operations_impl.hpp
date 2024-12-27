#ifndef CSR_OPERATIONS_IMPL_HPP
#define CSR_OPERATIONS_IMPL_HPP

#include "csr_operations.hpp"

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
    
}

#endif // CSR_OPERATIONS_IMPL_HPP