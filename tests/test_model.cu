// ============================================================================
// KernelFlow — Sequential Model Test
// ============================================================================
//
// Builds a DQN-style network for CartPole:
//   Input: 4 state values (cart pos, cart vel, pole angle, pole vel)
//   Output: 2 action probabilities (left, right)
//
// Architecture:
//   Linear(4 → 128) → ReLU → Linear(128 → 64) → ReLU →
//   Linear(64 → 2) → Softmax
//
// ============================================================================

#include "model.hpp"
#include <cstdio>
#include <cmath>

int main() {
    printf("============================================================\n");
    printf("  KernelFlow — Sequential Model Test (CartPole DQN)\n");
    printf("============================================================\n\n");

    // ---- Build the model ----
    Sequential model;
    model.add(new LinearLayerAdapter(4, 128));     // CartPole: 4 state inputs
    model.add(new ReLULayer());
    model.add(new LinearLayerAdapter(128, 64));
    model.add(new ReLULayer());
    model.add(new LinearLayerAdapter(64, 2));      // 2 actions: left, right
    model.add(new SoftmaxLayer());

    // Print model architecture
    model.print_summary();
    printf("\n");

    // ---- Create input: a single CartPole state ----
    // [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Tensor input({1, 4});
    input.cpu_ptr()[0] =  0.05f;   // cart slightly right
    input.cpu_ptr()[1] = -0.02f;   // cart moving slightly left
    input.cpu_ptr()[2] =  0.03f;   // pole tilted slightly right
    input.cpu_ptr()[3] =  0.01f;   // pole rotating slightly right
    input.to_gpu();

    printf("Input state: [%.3f, %.3f, %.3f, %.3f]\n",
           input.cpu_ptr()[0], input.cpu_ptr()[1],
           input.cpu_ptr()[2], input.cpu_ptr()[3]);

    // ---- Forward pass ----
    printf("Running forward pass...\n");
    Tensor output = model.forward(input);

    // ---- Read output on CPU ----
    output.to_cpu();

    printf("\nOutput action probabilities:\n");
    printf("  Left:  %.4f\n", output.cpu_ptr()[0]);
    printf("  Right: %.4f\n", output.cpu_ptr()[1]);
    printf("  Sum:   %.4f (should be ~1.0)\n",
           output.cpu_ptr()[0] + output.cpu_ptr()[1]);

    // ---- Verify softmax properties ----
    bool valid = true;
    float sum = 0.0f;
    for (int i = 0; i < output.size(); ++i) {
        float v = output.cpu_ptr()[i];
        if (v < 0.0f || v > 1.0f) valid = false;
        sum += v;
    }
    if (std::fabs(sum - 1.0f) > 1e-3f) valid = false;

    printf("\n[%s] Softmax output valid (all in [0,1], sum ≈ 1.0)\n",
           valid ? "PASS" : "FAIL");

    // ---- Batch forward pass test ----
    printf("\n--- Batch test: 32 states at once ---\n");
    Tensor batch_input({32, 4});
    batch_input.fill(0.1f);    // fill all states with 0.1

    Tensor batch_output = model.forward(batch_input);
    batch_output.to_cpu();

    printf("Batch output shape: ");
    batch_output.print_shape();

    auto shape = batch_output.get_shape();
    bool batch_ok = (shape[0] == 32 && shape[1] == 2);
    printf("[%s] Batch output shape is [32, 2]\n", batch_ok ? "PASS" : "FAIL");

    // Check first sample's probabilities sum to 1
    float s0 = batch_output.cpu_ptr()[0] + batch_output.cpu_ptr()[1];
    printf("[%s] First sample sums to %.4f (should be ~1.0)\n",
           std::fabs(s0 - 1.0f) < 1e-3f ? "PASS" : "FAIL", s0);

    printf("\n============================================================\n");
    printf("  Model test complete.\n");
    printf("============================================================\n");

    return 0;
}
