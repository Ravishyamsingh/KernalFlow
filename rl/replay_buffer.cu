// ============================================================================
// KernelFlow — GPU Experience Replay Buffer
// ============================================================================
//
// Stores RL transitions entirely on GPU memory. No host ↔ device copies
// during training — push and sample are pure GPU operations.
//
// Transition: (state, action, reward, next_state, done)
//
// Circular buffer: when capacity is reached, oldest transitions are
// overwritten. push_batch adds N transitions at once (from parallel envs).
// sample uses curand to generate random indices on GPU, then a gather
// kernel collects the sampled transitions.
//
// ============================================================================

#include "replay_buffer.hpp"

// ============================================================================
// Copy Batch Kernel — write a batch of transitions into the buffer
// ============================================================================
__global__ void copy_batch_kernel(
    float* buf_states, int* buf_actions, float* buf_rewards,
    float* buf_next_states, int* buf_dones,
    const float* src_states, const int* src_actions, const float* src_rewards,
    const float* src_next_states, const int* src_dones,
    int start_idx, int capacity, int state_dim, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int buf_idx = (start_idx + idx) % capacity;

        for (int d = 0; d < state_dim; ++d) {
            buf_states[buf_idx * state_dim + d] = src_states[idx * state_dim + d];
            buf_next_states[buf_idx * state_dim + d] = src_next_states[idx * state_dim + d];
        }

        buf_actions[buf_idx] = src_actions[idx];
        buf_rewards[buf_idx] = src_rewards[idx];
        buf_dones[buf_idx]   = src_dones[idx];
    }
}

// ============================================================================
// Random Index Kernel
// ============================================================================
__global__ void random_indices_kernel(int* indices, int batch_size,
                                      int max_idx, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        curandState rng;
        curand_init(seed, idx, 0, &rng);
        indices[idx] = curand(&rng) % max_idx;
    }
}

// ============================================================================
// Gather Kernel
// ============================================================================
__global__ void gather_kernel(
    const float* buf_states, const int* buf_actions, const float* buf_rewards,
    const float* buf_next_states, const int* buf_dones,
    const int* indices, int state_dim, int batch_size,
    float* out_states, int* out_actions, float* out_rewards,
    float* out_next_states, int* out_dones)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int src = indices[idx];

        for (int d = 0; d < state_dim; ++d) {
            out_states[idx * state_dim + d] = buf_states[src * state_dim + d];
            out_next_states[idx * state_dim + d] = buf_next_states[src * state_dim + d];
        }

        out_actions[idx] = buf_actions[src];
        out_rewards[idx] = buf_rewards[src];
        out_dones[idx]   = buf_dones[src];
    }
}

// ============================================================================
// ReplayBuffer — Method implementations
// ============================================================================

ReplayBuffer::ReplayBuffer(int capacity, int state_dim)
    : capacity_(capacity), state_dim_(state_dim),
      current_idx_(0), current_size_(0), sample_count_(0),
      d_indices_(nullptr), indices_capacity_(0)
{
    CHECK_CUDA(cudaMalloc(&d_states_,      capacity_ * state_dim_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_actions_,     capacity_ * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rewards_,     capacity_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_next_states_, capacity_ * state_dim_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dones_,       capacity_ * sizeof(int)));

    CHECK_CUDA(cudaMemset(d_states_,      0, capacity_ * state_dim_ * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_next_states_, 0, capacity_ * state_dim_ * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_actions_,     0, capacity_ * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_rewards_,     0, capacity_ * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dones_,       0, capacity_ * sizeof(int)));
}

ReplayBuffer::~ReplayBuffer() {
    if (d_states_)      cudaFree(d_states_);
    if (d_actions_)     cudaFree(d_actions_);
    if (d_rewards_)     cudaFree(d_rewards_);
    if (d_next_states_) cudaFree(d_next_states_);
    if (d_dones_)       cudaFree(d_dones_);
    if (d_indices_)     cudaFree(d_indices_);
}

void ReplayBuffer::push_batch(const float* d_src_states, const int* d_src_actions,
                              const float* d_src_rewards, const float* d_src_next_states,
                              const int* d_src_dones, int batch_size)
{
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    copy_batch_kernel<<<blocks, threads>>>(
        d_states_, d_actions_, d_rewards_, d_next_states_, d_dones_,
        d_src_states, d_src_actions, d_src_rewards,
        d_src_next_states, d_src_dones,
        current_idx_, capacity_, state_dim_, batch_size
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    current_idx_ = (current_idx_ + batch_size) % capacity_;
    current_size_ = min(current_size_ + batch_size, capacity_);
}

void ReplayBuffer::sample(int batch_size, float* out_states, int* out_actions,
                           float* out_rewards, float* out_next_states, int* out_dones)
{
    if (current_size_ < batch_size) {
        std::cerr << "ReplayBuffer: not enough transitions ("
                  << current_size_ << " < " << batch_size << ")" << std::endl;
        return;
    }

    if (batch_size > indices_capacity_) {
        if (d_indices_) cudaFree(d_indices_);
        CHECK_CUDA(cudaMalloc(&d_indices_, batch_size * sizeof(int)));
        indices_capacity_ = batch_size;
    }

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    sample_count_++;

    random_indices_kernel<<<blocks, threads>>>(
        d_indices_, batch_size, current_size_,
        42ULL + sample_count_
    );
    CHECK_CUDA(cudaGetLastError());

    gather_kernel<<<blocks, threads>>>(
        d_states_, d_actions_, d_rewards_, d_next_states_, d_dones_,
        d_indices_, state_dim_, batch_size,
        out_states, out_actions, out_rewards, out_next_states, out_dones
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}