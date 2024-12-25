#ifndef CSR_ADT_HPP
#define CSR_ADT_HPP

#include <vector>
#include <xtensor/xarray.hpp>
#include <iostream>
#include <xtensor/xstrided_view.hpp>

template <typename T>
class CSR
{
private:
    std::vector<T> values;                    // non zero values
    std::vector<std::vector<size_t>> indices; // indices of non zero values for each dimension
    std::vector<size_t> shape;                // shape of original tensor

public:
    // constructor for CSR using xarray or xtensor
    explicit CSR(const xt::xarray<T> &tensor)
    {
        shape = std::vector<size_t>(tensor.shape().begin().tensor.shape().end());

        for (auto i = tensor.begin(); i != tensor.end(); ++i)
        {
            if (*i != 0)
            {
                values.push_back(*i);

                auto flat_index = std::distance(tensor.begin(), i);
                std::vector multi_index(tensor.shape().size());
                auto = xt::strides(tensor.shape());
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
    const std::vector<T> &getValues() const
    {
        return values;
    }
    const std::vector<std::vector<size_t>> &getIndices() const
    {
        return indices;
    }
    const stf::vector<size_t> &getShape() const
    {
        return shape;
    }

    // Utilities
    void print() const
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
            for (auto j : indices[i])
            {
                std::cout << j << ", ";
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
};

#endif // CSR_ADT_HPP