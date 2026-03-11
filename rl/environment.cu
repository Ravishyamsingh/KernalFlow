// ============================================================================
// KernelFlow — Parallel CartPole Environment (GPU)
// ============================================================================
//
// Runs N environments simultaneously on GPU. Each thread handles one env.
//
// CartPole state: [cart_pos, cart_vel, pole_angle, pole_vel]
// Actions: 0 = push left (-10N), 1 = push right (+10N)
//
// Physics constants (same as OpenAI Gym CartPole-v1):
//   gravity     = 9.8 m/s²
//   cart_mass   = 1.0 kg
//   pole_mass   = 0.1 kg
//   pole_length = 0.5 m (half-length)
//   force_mag   = 10.0 N
//   dt          = 0.02 s
//
// Termination:
//   |pole_angle| > 12° (0.2094 rad) OR |cart_position| > 2.4
//
// ============================================================================

#include "environment.hpp"
#include <curand_kernel.h>

// Physics constants
#define GRAVITY       9.8f
#define CART_MASS     1.0f
#define POLE_MASS     0.1f
#define TOTAL_MASS    (CART_MASS + POLE_MASS)
#define POLE_LENGTH   0.5f
#define POLE_MASS_LEN (POLE_MASS * POLE_LENGTH)
#define FORCE_MAG     10.0f
#define DT            0.02f
#define ANGLE_LIMIT   0.2094395f   // 12 degrees in radians
#define CART_LIMIT    2.4f

// ============================================================================
// Reset Kernel — initialize all environments to small random states
// ============================================================================
__global__ void reset_envs_kernel(float* states, int n_envs,
                                  unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_envs) {
        curandState rng;
        curand_init(seed, idx, 0, &rng);

        // Small random values in [-0.05, 0.05] (same as Gym)
        float* s = states + idx * STATE_DIM;
        s[0] = (curand_uniform(&rng) - 0.5f) * 0.1f;  // cart_pos
        s[1] = (curand_uniform(&rng) - 0.5f) * 0.1f;  // cart_vel
        s[2] = (curand_uniform(&rng) - 0.5f) * 0.1f;  // pole_angle
        s[3] = (curand_uniform(&rng) - 0.5f) * 0.1f;  // pole_vel
    }
}

// ============================================================================
// Step Kernel — apply CartPole physics to all environments
// ============================================================================
//
// Uses semi-implicit Euler integration (same as OpenAI Gym):
//
//   force = action ? +10 : -10
//   cos_theta = cos(angle)
//   sin_theta = sin(angle)
//
//   temp = (force + pole_mass_length * angular_vel² * sin_theta) / total_mass
//   angular_accel = (gravity * sin_theta - cos_theta * temp) /
//                   (length * (4/3 - pole_mass * cos²_theta / total_mass))
//   cart_accel = temp - pole_mass_length * angular_accel * cos_theta / total_mass
//
//   cart_vel   += dt * cart_accel
//   cart_pos   += dt * cart_vel
//   pole_vel   += dt * angular_accel
//   pole_angle += dt * pole_vel
//
// ============================================================================
__global__ void step_envs_kernel(float* states, const int* actions,
                                 float* rewards, int* dones, int n_envs,
                                 unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_envs) {
        float* s = states + idx * STATE_DIM;

        float cart_pos   = s[0];
        float cart_vel   = s[1];
        float pole_angle = s[2];
        float pole_vel   = s[3];

        // Apply force based on action
        float force = (actions[idx] == 1) ? FORCE_MAG : -FORCE_MAG;

        float cos_theta = cosf(pole_angle);
        float sin_theta = sinf(pole_angle);

        // CartPole dynamics
        float temp = (force + POLE_MASS_LEN * pole_vel * pole_vel * sin_theta)
                     / TOTAL_MASS;
        float pole_accel = (GRAVITY * sin_theta - cos_theta * temp)
                           / (POLE_LENGTH * (4.0f / 3.0f - POLE_MASS * cos_theta
                              * cos_theta / TOTAL_MASS));
        float cart_accel = temp - POLE_MASS_LEN * pole_accel * cos_theta
                           / TOTAL_MASS;

        // Semi-implicit Euler integration
        cart_vel   += DT * cart_accel;
        cart_pos   += DT * cart_vel;
        pole_vel   += DT * pole_accel;
        pole_angle += DT * pole_vel;

        // Check termination
        int done = (fabsf(pole_angle) > ANGLE_LIMIT ||
                    fabsf(cart_pos) > CART_LIMIT) ? 1 : 0;

        // Reward: +1 for surviving, 0 if done
        rewards[idx] = done ? 0.0f : 1.0f;
        dones[idx] = done;

        // Write updated state
        s[0] = cart_pos;
        s[1] = cart_vel;
        s[2] = pole_angle;
        s[3] = pole_vel;

        // Auto-reset terminated environments to random initial state
        if (done) {
            curandState rng;
            curand_init(seed, idx + n_envs, 0, &rng);
            s[0] = (curand_uniform(&rng) - 0.5f) * 0.1f;
            s[1] = (curand_uniform(&rng) - 0.5f) * 0.1f;
            s[2] = (curand_uniform(&rng) - 0.5f) * 0.1f;
            s[3] = (curand_uniform(&rng) - 0.5f) * 0.1f;
        }
    }
}

// ============================================================================
// ParallelEnv — Host-side manager for parallel GPU environments
// ============================================================================

// ============================================================================
// ParallelEnv — Method implementations
// ============================================================================

ParallelEnv::ParallelEnv(int n_envs)
    : n_envs_(n_envs), d_states_(nullptr), d_rewards_(nullptr),
      d_dones_(nullptr), step_count_(0)
{
    CHECK_CUDA(cudaMalloc(&d_states_,  n_envs_ * STATE_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rewards_, n_envs_ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dones_,   n_envs_ * sizeof(int)));

    reset();
}

ParallelEnv::~ParallelEnv() {
    if (d_states_)  cudaFree(d_states_);
    if (d_rewards_) cudaFree(d_rewards_);
    if (d_dones_)   cudaFree(d_dones_);
}

void ParallelEnv::reset() {
    int threads = 256;
    int blocks = (n_envs_ + threads - 1) / threads;
    unsigned long long seed = 42ULL + step_count_;

    reset_envs_kernel<<<blocks, threads>>>(d_states_, n_envs_, seed);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void ParallelEnv::step(int* d_actions) {
    int threads = 256;
    int blocks = (n_envs_ + threads - 1) / threads;
    step_count_++;
    unsigned long long seed = 42ULL + step_count_;

    step_envs_kernel<<<blocks, threads>>>(
        d_states_, d_actions, d_rewards_, d_dones_, n_envs_, seed
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}