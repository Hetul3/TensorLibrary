cmake_minimum_required(VERSION 3.10)
project(sparse_lib)

# xtensor
find_package(xtensor REQUIRED)

# Include directories
include_directories(include)

# Library target
add_library(sparse_ops src/sparse_operations.cpp)
target_link_libraries(sparse_ops xtensor)

# Test executable
add_executable(test_sparse_operations tests/test_sparse_operations.cpp)
target_link_libraries(test_sparse_operations sparse_ops xtensor)