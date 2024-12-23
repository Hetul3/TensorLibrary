#include "sparse_operations.hpp"
#include <algorithm>

namespace sparse_ops {
    template <typename Tensor>
    bool isSparse(const Tensor& tensor, double threshold) {
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
    double sparsity(const Tensor& tensor) {
       auto zero_count = xt::sum(xt::equal(tensor, 0.0));
       return static_cast<double>(zero_count()) / tensor.size();
    }

    template bool isSparse<xt::xarray<double>>(const xt::xarray<double>&, double);
    template bool isSparse<xt::xtensor<double, xt::dynamic_shape>>(const xt::xtensor<double, xt::dynamic_shape>&, double);
}