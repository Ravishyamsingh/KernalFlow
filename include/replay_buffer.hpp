#pragma once

#include "tensor.hpp"
#include <curand_kernel.h>
#include <iostream>

// Kernels (defined in rl/replay_buffer.cu)
__global__ void copy_batch_kernel(
    float* buf_states, int* buf_actions, float* buf_rewards,
    float* buf_next_states, int* buf_dones,
    const float* src_states, const int* src_actions, const float* src_rewards,
    const float* src_next_states, const int* src_dones,
    int start_idx, int capacity, int state_dim, int batch_size);

__global__ void random_indices_kernel(int* indices, int batch_size,
                                      int max_idx, unsigned long long seed);

__global__ void gather_kernel(
    const float* buf_states, const int* buf_actions, const float* buf_rewards,
    const float* buf_next_states, const int* buf_dones,
    const int* indices, int state_dim, int batch_size,
    float* out_states, int* out_actions, float* out_rewards,
    float* out_next_states, int* out_dones);

class ReplayBuffer {
private:
    int capacity_;
    int state_dim_;
    int current_idx_;
    int current_size_;
    unsigned long long sample_count_;

    float*  d_states_;
    int*    d_actions_;
    float*  d_rewards_;
    float*  d_next_states_;
    int*    d_dones_;

    int*    d_indices_;
    int     indices_capacity_;

public:
    ReplayBuffer(int capacity = 100000, int state_dim = 4);
    ~ReplayBuffer();

    ReplayBuffer(const ReplayBuffer&) = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    void push_batch(const float* d_src_states, const int* d_src_actions,
                    const float* d_src_rewards, const float* d_src_next_states,
                    const int* d_src_dones, int batch_size);

    void sample(int batch_size, float* out_states, int* out_actions,
                float* out_rewards, float* out_next_states, int* out_dones);

    int size() const { return current_size_; }
    bool ready(int min_size) const { return current_size_ >= min_size; }
};
