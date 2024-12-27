#ifndef CSR_ADT_IMPL_HPP
#define CSR_ADT_IMPL_HPP

#include "csr_adt.hpp"
#include <xtensor/xstrided_view.hpp>
#include <iostream>

// Constructor
template <typename T>
CSR<T>::CSR(const xt::xarray<T> &tensor)
{
    shape = std::vector<size_t>(tensor.shape().begin().tensor.shape().end());

    for (auto i = tensor.begin(); i != tensor.end(); ++i)
    {
        if (*i != 0)
        {
            values.push_back(*i);

            auto flat_index = std::distance(tensor.begin(), i);
            std::vector multi_index(tensor.shape().size());
            auto strides = xt::strides(tensor.shape());
            for (auto dim = 0; dim < tensor.shape().size(); ++dim)
            {
                multi_index[dim] = flat_index / strides[dim];
                flat_index %= strides[dim];
            }
            indices.push_back(multi_index);
        }
    }
}

// Accessors
template <typename T>
const std::vector<T> &CSR<T>::getValues() const
{
    return values;
}
template <typename T>
const std::vector<std::vector<size_t>> &CSR<T>::getIndices() const
{
    return indices;
}
template <typename T>
const std::vector<size_t> &CSR<T>::getShape() const
{
    return shape;
}

// Utilities
template <typename T>
void CSR<T>::print() const
{
    std::cout << "Shape: [";
    for (auto i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i != shape.size() - 1)
        {
            std::cout << ", ";
        }
        else
        {
            std::cout << "]";
        }
    }
    std::cout << std::endl;
    std::cout << "(Values : [Indices]): ";
    for (auto i = 0; i < values.size(); ++i)
    {
        std::cout << "(" << values[i] << " : [";
        for (auto j = 0; j < indices[i].size(); ++j)
        {
            std::cout << indices[i][j];
            if (j != indices[i].size() - 1)
            {
                std::cout << ", ";
            }
        }
        if (i != values.size() - 1)
        {
            std::cout << "]), ";
        }
        else
        {
            std::cout << "])";
        }
    }
    std::cout << std::endl;
}

#endif // CSR_ADT_IMPL_HPP