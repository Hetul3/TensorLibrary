#ifndef CSR_ADT_HPP
#define CSR_ADT_HPP

#include <vector>
#include <xtensor/xarray.hpp>

template <typename T>
class CSR
{
private:
    std::vector<T> values;                    // non zero values
    std::vector<std::vector<size_t>> indices; // indices of non zero values for each dimension
    std::vector<size_t> shape;                // shape of original tensor

public:
    // constructor for CSR using xarray or xtensor
    explicit CSR(const xt::xarray<T> &tensor);

    // Accessors
    const std::vector<T> &getValues() const;
    const std::vector<std::vector<size_t>> &getIndices() const;
    const std::vector<size_t> &getShape() const;

    // Utilities
    void print() const;
};

#include "csr_adt_impl.hpp"

#endif // CSR_ADT_HPP