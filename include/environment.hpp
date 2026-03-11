#pragma once

#include "tensor.hpp"

#define STATE_DIM 4

// Kernels (defined in rl/environment.cu)
__global__ void reset_envs_kernel(float* states, int n_envs, unsigned long long seed);
__global__ void step_envs_kernel(float* states, const int* actions, float* rewards,
                                 int* dones, int n_envs, unsigned long long seed);

class ParallelEnv {
private:
    int n_envs_;
    float* d_states_;
    float* d_rewards_;
    int*   d_dones_;
    unsigned long long step_count_;

public:
    ParallelEnv(int n_envs = 512);
    ~ParallelEnv();

    ParallelEnv(const ParallelEnv&) = delete;
    ParallelEnv& operator=(const ParallelEnv&) = delete;

    void reset();
    void step(int* d_actions);

    float* get_states()  { return d_states_; }
    float* get_rewards() { return d_rewards_; }
    int*   get_dones()   { return d_dones_; }
    int    get_n_envs()  { return n_envs_; }
};
