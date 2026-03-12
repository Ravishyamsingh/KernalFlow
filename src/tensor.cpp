#include "tensor.hpp"

// ============================================================================
// GPU Kernel: fill_kernel
// Each thread writes one element. Standard 1D grid-stride pattern.
// ============================================================================
__global__ void fill_kernel(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// ============================================================================
// Constructor
// Allocates zeroed CPU memory (new float[]) and GPU memory (cudaMalloc).
// Both are allocated upfront so to_gpu()/to_cpu() are pure memcpy — no
// allocation latency during inference.
// ============================================================================
Tensor::Tensor(std::vector<int> shape)
    : d_data(nullptr),
      h_data(nullptr),
      shape_(std::move(shape)),
      total_size_(0),
      on_gpu_(false)

{
    // Compute total element count by multiplying all dimensions
    total_size_ = std::accumulate(shape_.begin(), shape_.end(), 1,
                                  std::multiplies<int>());

    // Allocate zeroed host memory
    h_data = new float[total_size_]();

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_data, total_size_ * sizeof(float)));
}

// ============================================================================
// Move Constructor
// Transfers ownership of both CPU and GPU memory from source tensor.
// Source is left in a valid empty state.
// ============================================================================
Tensor::Tensor(Tensor&& other) noexcept
    : d_data(other.d_data),
      h_data(other.h_data),
      shape_(std::move(other.shape_)),
      total_size_(other.total_size_),
      on_gpu_(other.on_gpu_)
{
    other.d_data = nullptr;
    other.h_data = nullptr;
    other.total_size_ = 0;
}

// ============================================================================
// Move Assignment Operator
// Frees current resources, then takes ownership from source.
// ============================================================================
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        delete[] h_data;
        if (d_data) cudaFree(d_data);

        // Take ownership
        d_data = other.d_data;
        h_data = other.h_data;
        shape_ = std::move(other.shape_);
        total_size_ = other.total_size_;
        on_gpu_ = other.on_gpu_;

        // Nullify source
        other.d_data = nullptr;
        other.h_data = nullptr;
        other.total_size_ = 0;
    }
    return *this;
}

// ============================================================================
// to_gpu — Copy CPU data to GPU
// Uses cudaMemcpyHostToDevice. Assumes both h_data and d_data are allocated.
// ============================================================================
void Tensor::to_gpu() {
    CHECK_CUDA(cudaMemcpy(d_data, h_data,
                          total_size_ * sizeof(float),
                          cudaMemcpyHostToDevice));
    on_gpu_ = true;
}

// ============================================================================
// to_cpu — Copy GPU data back to CPU
// Uses cudaMemcpyDeviceToHost. Call this before reading h_data after GPU ops.
// ============================================================================
void Tensor::to_cpu() {
    CHECK_CUDA(cudaMemcpy(h_data, d_data,
                          total_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost));
    on_gpu_ = false;
}

// ============================================================================
// fill — Fill all elements with a constant value on GPU
// Launches fill_kernel with 256 threads per block.
// Synchronizes after launch to ensure completion before returning.
// ============================================================================
void Tensor::fill(float value) {
    int threads = 256;
    int blocks = (total_size_ + threads - 1) / threads;

    fill_kernel<<<blocks, threads>>>(d_data, value, total_size_);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    on_gpu_ = true;
}

// ============================================================================
// Pointer accessors — return raw pointers for kernel launches
// ============================================================================
float* Tensor::gpu_ptr() { return d_data; }
float* Tensor::cpu_ptr() { return h_data; }

// ============================================================================
// Shape and size queries
// ============================================================================
std::vector<int> Tensor::get_shape() const { return shape_; }
int Tensor::size() const { return total_size_; }
bool Tensor::is_on_gpu() const { return on_gpu_; }

// ============================================================================
// print_shape — Prints dimensions to console
// Example output: "Tensor[32, 256]"
// ============================================================================
void Tensor::print_shape() const {
    std::cout << "Tensor[";
    for (int i = 0; i < static_cast<int>(shape_.size()); ++i) {
        std::cout << shape_[i];
        if (i < static_cast<int>(shape_.size()) - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// ============================================================================
// Destructor — Frees both CPU and GPU memory
// cudaFree is safe to call with nullptr, but we guard anyway for clarity.
// ============================================================================
Tensor::~Tensor() {
    delete[] h_data;
    h_data = nullptr;

    if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
}