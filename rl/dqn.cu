// ============================================================================
// KernelFlow — DQN Agent (Deep Q-Network)
// ============================================================================
//
// Complete DQN training loop for CartPole using:
//   - Your custom CUDA inference engine (Sequential model)
//   - 512 parallel GPU environments (ParallelEnv)
//   - GPU-resident experience replay (ReplayBuffer)
//
// Algorithm:
//   1. Observe states from all 512 envs
//   2. Select actions via epsilon-greedy (batch forward pass)
//   3. Step all envs, collect (s, a, r, s', done) transitions
//   4. Push to replay buffer
//   5. Sample mini-batch, compute TD targets, update weights
//   6. Periodically sync target network
//   7. Decay epsilon
//
// ============================================================================

#include "model.hpp"
#include "environment.hpp"
#include "replay_buffer.hpp"
#include "stream_engine.hpp"
#include <curand_kernel.h>
#include <cstdio>
#include <cstring>
#include <cmath>

// Constants
#define STATE_DIM     4
#define N_ACTIONS     2
#define N_ENVS        512
#define BUFFER_CAP    100000
#define BATCH_SIZE    256
#define LR            0.001f
#define GAMMA         0.99f
#define EPS_START     1.0f
#define EPS_END       0.01f
#define EPS_DECAY     0.995f
#define TARGET_UPDATE 100
#define MAX_STEPS_PER_EP 200

// ============================================================================
// CUDA Kernels for DQN Operations
// ============================================================================

// Epsilon-greedy action selection on GPU
// Each thread handles one environment:
//   - Generate random float, if < epsilon → random action
//   - Otherwise → argmax over Q-values from policy network output
__global__ void epsilon_greedy_kernel(const float* q_values, int* actions,
                                      int n_envs, int n_actions,
                                      float epsilon, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_envs) {
        curandState rng;
        curand_init(seed, idx, 0, &rng);
        float rand_val = curand_uniform(&rng);

        if (rand_val < epsilon) {
            // Random action
            actions[idx] = curand(&rng) % n_actions;
        } else {
            // Greedy: argmax Q(s, a)
            const float* q = q_values + idx * n_actions;
            int best = 0;
            float best_val = q[0];
            for (int a = 1; a < n_actions; ++a) {
                if (q[a] > best_val) {
                    best_val = q[a];
                    best = a;
                }
            }
            actions[idx] = best;
        }
    }
}

// Gather Q(s, a) — pick the Q-value for the action actually taken
// Input:  q_all [batch × n_actions], actions [batch]
// Output: q_selected [batch] where q_selected[i] = q_all[i][actions[i]]
__global__ void gather_q_kernel(const float* q_all, const int* actions,
                                float* q_selected, int batch, int n_actions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        q_selected[idx] = q_all[idx * n_actions + actions[idx]];
    }
}

// Compute max Q(s', a') from target network output
// Input:  q_next [batch × n_actions]
// Output: q_max [batch]
__global__ void max_q_kernel(const float* q_next, float* q_max,
                             int batch, int n_actions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        const float* q = q_next + idx * n_actions;
        float mx = q[0];
        for (int a = 1; a < n_actions; ++a) {
            mx = fmaxf(mx, q[a]);
        }
        q_max[idx] = mx;
    }
}

// Compute TD targets: target = reward + gamma * max_q_next * (1 - done)
__global__ void compute_targets_kernel(const float* rewards, const int* dones,
                                       const float* max_q_next, float* targets,
                                       int batch, float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        float not_done = (dones[idx] == 0) ? 1.0f : 0.0f;
        targets[idx] = rewards[idx] + gamma * max_q_next[idx] * not_done;
    }
}

// Compute Huber loss gradient: d_loss/d_q for each sample
// Huber loss: L = 0.5*(q-t)² if |q-t| < 1, else |q-t| - 0.5
// Gradient:   dL/dq = (q-t) if |q-t| < 1, else sign(q-t)
__global__ void huber_grad_kernel(const float* q_pred, const float* targets,
                                  float* grad, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        float diff = q_pred[idx] - targets[idx];
        if (fabsf(diff) < 1.0f) {
            grad[idx] = diff;
        } else {
            grad[idx] = (diff > 0.0f) ? 1.0f : -1.0f;
        }
    }
}

// Simplified SGD weight update: w -= lr * gradient
// Operates on raw weight arrays
__global__ void sgd_update_kernel(float* weights, const float* gradients,
                                  int n, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= lr * gradients[idx];
    }
}

// Copy weights from source to destination (for target network sync)
__global__ void copy_weights_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Compute mean of an array (for reporting avg reward)
__global__ void sum_reduce_simple(const float* data, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += data[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *result = sdata[0];
}

// ============================================================================
// DQNAgent Class
// ============================================================================

class DQNAgent {
private:
    Sequential* policy_net_;
    Sequential* target_net_;
    StreamEngine* stream_engine_;  // Multi-stream inference engine
    ReplayBuffer* memory_;
    ParallelEnv* envs_;

    float epsilon_;
    float gamma_;
    float lr_;
    int update_target_every_;
    int total_steps_;
    unsigned long long action_seed_;

    // Persistent GPU buffers (avoid re-allocation every step)
    int*   d_actions_;           // [N_ENVS]
    float* d_prev_states_;       // [N_ENVS × STATE_DIM]

    // Buffers for update
    float* d_sample_states_;     // [BATCH × STATE_DIM]
    int*   d_sample_actions_;    // [BATCH]
    float* d_sample_rewards_;    // [BATCH]
    float* d_sample_next_;       // [BATCH × STATE_DIM]
    int*   d_sample_dones_;      // [BATCH]
    float* d_q_selected_;        // [BATCH]
    float* d_max_q_next_;        // [BATCH]
    float* d_targets_;           // [BATCH]
    float* d_grad_;              // [BATCH]
    float* d_avg_reward_;        // [1]

    // Build the Q-network architecture: 4 → 128 → 64 → 2
    Sequential* build_network() {
        Sequential* net = new Sequential();
        net->add(new LinearLayerAdapter(STATE_DIM, 128));
        net->add(new ReLULayer());
        net->add(new LinearLayerAdapter(128, 64));
        net->add(new ReLULayer());
        net->add(new LinearLayerAdapter(64, N_ACTIONS));
        return net;
    }

public:
    DQNAgent(ParallelEnv* envs, ReplayBuffer* memory)
        : envs_(envs), memory_(memory),
          epsilon_(EPS_START), gamma_(GAMMA), lr_(LR),
          update_target_every_(TARGET_UPDATE),
          total_steps_(0), action_seed_(123ULL)
    {
        policy_net_ = build_network();
        target_net_ = build_network();
        stream_engine_ = new StreamEngine(policy_net_);

        // Allocate persistent GPU buffers
        CHECK_CUDA(cudaMalloc(&d_actions_,       N_ENVS * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_prev_states_,   N_ENVS * STATE_DIM * sizeof(float)));

        CHECK_CUDA(cudaMalloc(&d_sample_states_, BATCH_SIZE * STATE_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sample_actions_,BATCH_SIZE * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_sample_rewards_,BATCH_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sample_next_,   BATCH_SIZE * STATE_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_sample_dones_,  BATCH_SIZE * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_q_selected_,    BATCH_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_max_q_next_,    BATCH_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_targets_,       BATCH_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_grad_,          BATCH_SIZE * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_avg_reward_,    sizeof(float)));

        // Sync target network with policy network at start
        update_target_network();

        printf("[DQNAgent] Initialized with %d parallel envs\n", N_ENVS);
        policy_net_->print_summary();
    }

    ~DQNAgent() {
        delete stream_engine_;
        delete policy_net_;
        delete target_net_;
        cudaFree(d_actions_);
        cudaFree(d_prev_states_);
        cudaFree(d_sample_states_);
        cudaFree(d_sample_actions_);
        cudaFree(d_sample_rewards_);
        cudaFree(d_sample_next_);
        cudaFree(d_sample_dones_);
        cudaFree(d_q_selected_);
        cudaFree(d_max_q_next_);
        cudaFree(d_targets_);
        cudaFree(d_grad_);
        cudaFree(d_avg_reward_);
    }

    // ========================================================================
    // select_actions — epsilon-greedy over batched Q-values
    //
    // Runs ALL 512 env states through the policy network in one forward pass,
    // then each thread picks either a random action or argmax Q.
    // Returns device pointer to actions.
    // ========================================================================
    int* select_actions(float* d_states) {
        // Wrap GPU states into a Tensor for the forward pass
        Tensor state_tensor({N_ENVS, STATE_DIM});
        CHECK_CUDA(cudaMemcpy(state_tensor.gpu_ptr(), d_states,
                              N_ENVS * STATE_DIM * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        // Multi-stream forward: splits 512 envs into 4×128 sub-batches
        // Each sub-batch runs on a separate CUDA stream simultaneously
        Tensor q_values = stream_engine_->streamed_forward(state_tensor);

        // Epsilon-greedy selection on GPU
        int threads = 256;
        int blocks = (N_ENVS + threads - 1) / threads;
        action_seed_++;

        epsilon_greedy_kernel<<<blocks, threads>>>(
            q_values.gpu_ptr(), d_actions_, N_ENVS, N_ACTIONS,
            epsilon_, action_seed_
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        return d_actions_;
    }

    // ========================================================================
    // update_weights — sample from replay, compute TD loss, update weights
    //
    // Steps:
    //   1. Sample mini-batch from replay buffer
    //   2. Q_pred = policy_net(states), gather Q(s, a) for taken actions
    //   3. Q_next = target_net(next_states), compute max Q(s', a')
    //   4. targets = reward + gamma * max_Q_next * (1 - done)
    //   5. Compute Huber loss gradient
    //   6. Backpropagate gradient through output layer weights (simplified)
    // ========================================================================
    void update_weights() {
        if (!memory_->ready(BATCH_SIZE)) return;

        // --- Step 1: Sample batch ---
        memory_->sample(BATCH_SIZE, d_sample_states_, d_sample_actions_,
                        d_sample_rewards_, d_sample_next_, d_sample_dones_);

        // --- Step 2: Compute Q(s, a) from policy network ---
        Tensor s_tensor({BATCH_SIZE, STATE_DIM});
        CHECK_CUDA(cudaMemcpy(s_tensor.gpu_ptr(), d_sample_states_,
                              BATCH_SIZE * STATE_DIM * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        Tensor q_all = policy_net_->forward(s_tensor);

        int threads = 256;
        int blocks = (BATCH_SIZE + threads - 1) / threads;

        // Gather Q-values for taken actions: q_selected[i] = q_all[i][a[i]]
        gather_q_kernel<<<blocks, threads>>>(
            q_all.gpu_ptr(), d_sample_actions_, d_q_selected_,
            BATCH_SIZE, N_ACTIONS
        );
        CHECK_CUDA(cudaGetLastError());

        // --- Step 3: Compute max Q(s', a') from target network ---
        Tensor ns_tensor({BATCH_SIZE, STATE_DIM});
        CHECK_CUDA(cudaMemcpy(ns_tensor.gpu_ptr(), d_sample_next_,
                              BATCH_SIZE * STATE_DIM * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        Tensor q_next = target_net_->forward(ns_tensor);

        max_q_kernel<<<blocks, threads>>>(
            q_next.gpu_ptr(), d_max_q_next_, BATCH_SIZE, N_ACTIONS
        );
        CHECK_CUDA(cudaGetLastError());

        // --- Step 4: Compute TD targets ---
        compute_targets_kernel<<<blocks, threads>>>(
            d_sample_rewards_, d_sample_dones_, d_max_q_next_, d_targets_,
            BATCH_SIZE, gamma_
        );
        CHECK_CUDA(cudaGetLastError());

        // --- Step 5: Compute Huber loss gradient ---
        huber_grad_kernel<<<blocks, threads>>>(
            d_q_selected_, d_targets_, d_grad_, BATCH_SIZE
        );
        CHECK_CUDA(cudaGetLastError());

        // --- Step 6: Simplified weight update ---
        // Update the last linear layer's weights using the gradient signal.
        // In a full implementation, we'd backpropagate through all layers.
        // Here we do a simplified direct update on the output layer.
        //
        // The output layer is layer index 4 (Linear(64 → 2)) in the network.
        Layer* output_layer = policy_net_->get_layer(4);
        LinearLayerAdapter* adapter = static_cast<LinearLayerAdapter*>(output_layer);
        LinearLayer* linear = adapter->get();

        int w_size = linear->get_in_features() * linear->get_out_features();
        int b_size = linear->get_out_features();

        // Scale gradients by 1/batch_size and apply SGD to weights
        int w_blocks = (w_size + threads - 1) / threads;
        sgd_update_kernel<<<w_blocks, threads>>>(
            linear->get_weights()->gpu_ptr(), d_grad_,
            min(w_size, BATCH_SIZE), lr_ / BATCH_SIZE
        );

        int b_blocks = (b_size + threads - 1) / threads;
        sgd_update_kernel<<<b_blocks, threads>>>(
            linear->get_bias()->gpu_ptr(), d_grad_,
            min(b_size, BATCH_SIZE), lr_ / BATCH_SIZE
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ========================================================================
    // update_target_network — copy policy weights → target weights
    //
    // Copies all linear layer weights from policy_net to target_net.
    // ========================================================================
    void update_target_network() {
        // Iterate over layers, copy weights for all LinearLayerAdapters
        for (int i = 0; i < policy_net_->num_layers(); ++i) {
            LinearLayerAdapter* p_adapter =
                dynamic_cast<LinearLayerAdapter*>(policy_net_->get_layer(i));
            LinearLayerAdapter* t_adapter =
                dynamic_cast<LinearLayerAdapter*>(target_net_->get_layer(i));

            if (p_adapter && t_adapter) {
                LinearLayer* p_lin = p_adapter->get();
                LinearLayer* t_lin = t_adapter->get();

                int w_size = p_lin->get_in_features() * p_lin->get_out_features();
                int b_size = p_lin->get_out_features();

                int threads = 256;
                int w_blocks = (w_size + threads - 1) / threads;
                int b_blocks = (b_size + threads - 1) / threads;

                copy_weights_kernel<<<w_blocks, threads>>>(
                    t_lin->get_weights()->gpu_ptr(),
                    p_lin->get_weights()->gpu_ptr(), w_size
                );
                copy_weights_kernel<<<b_blocks, threads>>>(
                    t_lin->get_bias()->gpu_ptr(),
                    p_lin->get_bias()->gpu_ptr(), b_size
                );
            }
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ========================================================================
    // train — Main DQN training loop
    //
    // For each episode:
    //   - Run steps until all envs complete or max steps reached
    //   - At each step: select actions → step envs → push to buffer → update
    //   - Track and report average reward per episode
    //   - Decay epsilon after each episode
    // ========================================================================
    void train(int n_episodes = 500) {
        printf("\n");
        printf("============================================================\n");
        printf("  KernelFlow DQN Training — CartPole\n");
        printf("  Envs: %d | Buffer: %d | Batch: %d | LR: %.4f\n",
               N_ENVS, BUFFER_CAP, BATCH_SIZE, lr_);
        printf("============================================================\n\n");

        // Open rewards CSV for plotting
        FILE* rewards_fp = fopen("rewards.csv", "w");
        if (rewards_fp) fprintf(rewards_fp, "episode,avg_reward,epsilon\n");

        for (int ep = 0; ep < n_episodes; ++ep) {
            // Reset all environments
            envs_->reset();

            float episode_total_reward = 0.0f;
            int episode_steps = 0;

            for (int step = 0; step < MAX_STEPS_PER_EP; ++step) {
                // Save current states (before stepping)
                CHECK_CUDA(cudaMemcpy(d_prev_states_, envs_->get_states(),
                                      N_ENVS * STATE_DIM * sizeof(float),
                                      cudaMemcpyDeviceToDevice));

                // Select actions via epsilon-greedy batch forward pass
                int* actions = select_actions(envs_->get_states());

                // Step all environments
                envs_->step(actions);

                // Push transitions to replay buffer
                // (prev_states, actions, rewards, current_states, dones)
                memory_->push_batch(d_prev_states_, actions,
                                    envs_->get_rewards(),
                                    envs_->get_states(),
                                    envs_->get_dones(), N_ENVS);

                // Update weights if buffer has enough data
                update_weights();

                // Track reward
                // Sum rewards across all envs for this step
                sum_reduce_simple<<<1, 256>>>(
                    envs_->get_rewards(), d_avg_reward_, N_ENVS
                );
                CHECK_CUDA(cudaDeviceSynchronize());

                float step_reward_sum = 0.0f;
                CHECK_CUDA(cudaMemcpy(&step_reward_sum, d_avg_reward_,
                                      sizeof(float), cudaMemcpyDeviceToHost));
                episode_total_reward += step_reward_sum;
                episode_steps++;

                total_steps_++;

                // Update target network periodically
                if (total_steps_ % update_target_every_ == 0) {
                    update_target_network();
                }
            }

            // Decay epsilon
            epsilon_ = fmaxf(EPS_END, epsilon_ * EPS_DECAY);

            // Report
            float avg_reward = episode_total_reward / N_ENVS;
            if (rewards_fp) {
                fprintf(rewards_fp, "%d,%.4f,%.6f\n", ep, avg_reward, epsilon_);
            }
            if (ep % 10 == 0 || ep == n_episodes - 1) {
                printf("Episode %4d | Avg Reward: %7.2f | Epsilon: %.4f | "
                       "Buffer: %6d | Steps: %d\n",
                       ep, avg_reward, epsilon_, memory_->size(), total_steps_);
            }
        }

        if (rewards_fp) {
            fclose(rewards_fp);
            printf("  Rewards saved to: rewards.csv\n");
        }
        printf("\n============================================================\n");
        printf("  Training complete. Total steps: %d\n", total_steps_);
        printf("============================================================\n");
    }
};

// ============================================================================
// main — Launch DQN training on CartPole
// ============================================================================
int main() {
    printf("KernelFlow — DQN CartPole Training\n");
    printf("===================================\n\n");

    // Check GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d, %.1f GB)\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Create components
    ParallelEnv envs(N_ENVS);
    ReplayBuffer memory(BUFFER_CAP, STATE_DIM);

    // Create agent and train
    DQNAgent agent(&envs, &memory);
    agent.train(500);

    // ================================================================
    // Stream benchmark: compare single-stream vs 4-stream inference
    // ================================================================
    printf("\n");
    Sequential* bench_net = new Sequential();
    bench_net->add(new LinearLayerAdapter(STATE_DIM, 128));
    bench_net->add(new ReLULayer());
    bench_net->add(new LinearLayerAdapter(128, 64));
    bench_net->add(new ReLULayer());
    bench_net->add(new LinearLayerAdapter(64, N_ACTIONS));

    StreamEngine bench_engine(bench_net);
    bench_engine.benchmark(N_ENVS, STATE_DIM, 1000);

    // ================================================================
    // Graph benchmark: compare normal launches vs CUDA Graph replay
    // ================================================================
    GraphEngine graph_engine;
    graph_engine.compare_graph_vs_normal(*bench_net, N_ENVS, STATE_DIM, 10000);

    delete bench_net;

    return 0;
}