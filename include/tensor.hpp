#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <numeric>

// CUDA error checking macro — aborts on failure with file/line info
#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

class Tensor {
private:
    float* d_data;              // Device (GPU) pointer
    float* h_data;              // Host (CPU) pointer
    std::vector<int> shape_;    // Dimensions, e.g. {batch, channels, H, W}
    int total_size_;            // Total number of elements
    bool on_gpu_;               // Tracks whether the latest data lives on GPU

public:
    // Construct a tensor with the given shape, allocates both CPU and GPU memory
    explicit Tensor(std::vector<int> shape);

    // No copies — tensor owns its memory
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Copy host data to device (CPU → GPU)
    void to_gpu();

    // Copy device data back to host (GPU → CPU)
    void to_cpu();

    // Fill all elements with a value using a GPU kernel
    void fill(float value);

    // Raw pointer accessors
    float* gpu_ptr();
    float* cpu_ptr();

    // Shape and size queries
    std::vector<int> get_shape() const;
    int size() const;
    bool is_on_gpu() const;

    // Print shape to console, e.g. "Tensor[32, 256]"
    void print_shape() const;

    // Free both CPU and GPU memory
    ~Tensor();
};