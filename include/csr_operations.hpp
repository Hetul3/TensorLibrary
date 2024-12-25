#ifndef CSR_OPERATIONS_HPP
#define CSR_OPERATIONS_HPP

#include "csr_adt.hpp"

// convert CSR object to an xarray
template <typename T>
xt::xarray<T> CSRToDense(const CSR<T>& csr);

// convert xarray to CSR object
template <typename T>
CSR<T> DenseToCSR(const xt::xarray<T>& tensor);

// multiply two CSR objects
template <typename T>
CSR<T> CSRMult(const CSR<T>& csr1, const CSR<T>& csr2);

#endif // CSR_OPERATIONS.HPP