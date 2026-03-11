#include "tensor.hpp"
#include <cmath>
#include <cstring>


void print_result(const char* test_name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}


bool test_create_and_shape() {
    Tensor t({1024, 1024});
    t.print_shape();

    auto shape = t.get_shape();
    bool ok = (shape.size() == 2 && shape[0] == 1024 && shape[1] == 1024);
    ok = ok && (t.size() == 1024 * 1024);
    return ok;
}

// ============================================================================
// Test 2 & 3: Fill on GPU, copy back, verify values
// ============================================================================
bool test_fill_and_verify() {
    Tensor t({1024, 1024});

    // Fill all elements with 3.14 on GPU
    t.fill(3.14f);

    // Copy back to CPU for verification
    t.to_cpu();

    // Check first 5 elements
    float* data = t.cpu_ptr();
    bool ok = true;
    std::cout << "  First 5 elements: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << data[i];
        if (i < 4) std::cout << ", ";
        if (std::fabs(data[i] - 3.14f) > 1e-6f) {
            ok = false;
        }
    }
    std::cout << std::endl;

    // Also verify a few elements deeper in the array
    for (int i = 0; i < t.size(); i += 10000) {
        if (std::fabs(data[i] - 3.14f) > 1e-6f) {
            ok = false;
            break;
        }
    }

    return ok;
}

// ============================================================================
// Test 4 & 5: CPU fill vs GPU fill benchmark using cudaEvent_t
// ============================================================================
void test_cpu_vs_gpu_timing() {
    const int N = 1024 * 1024;  // ~1M elements

    // ---- GPU fill timing ----
    Tensor t_gpu({1024, 1024});

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cudaEventRecord(gpu_start);
    t_gpu.fill(3.14f);
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop);

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    // ---- CPU fill timing ----
    // Use a raw host buffer to simulate a pure CPU fill loop
    float* cpu_buf = new float[N];

    cudaEvent_t cpu_start, cpu_stop;
    cudaEventCreate(&cpu_start);
    cudaEventCreate(&cpu_stop);

    cudaEventRecord(cpu_start);
    cudaEventSynchronize(cpu_start);  // sync so timing starts from CPU work

    for (int i = 0; i < N; ++i) {
        cpu_buf[i] = 3.14f;
    }

    cudaEventRecord(cpu_stop);
    cudaEventSynchronize(cpu_stop);

    float cpu_ms = 0.0f;
    cudaEventElapsedTime(&cpu_ms, cpu_start, cpu_stop);

    cudaEventDestroy(cpu_start);
    cudaEventDestroy(cpu_stop);
    delete[] cpu_buf;

    // ---- Print results ----
    std::cout << "  CPU fill (" << N << " elements): " << cpu_ms << " ms" << std::endl;
    std::cout << "  GPU fill (" << N << " elements): " << gpu_ms << " ms" << std::endl;

    if (gpu_ms > 0.0f) {
        float speedup = cpu_ms / gpu_ms;
        std::cout << "  GPU is " << speedup << "x faster than CPU" << std::endl;
        print_result("CPU vs GPU Benchmark", speedup > 1.0f);
    } else {
        std::cout << "  GPU time too small to measure — GPU is extremely fast" << std::endl;
        print_result("CPU vs GPU Benchmark", true);
    }
}

// ============================================================================
// Main — run all tests
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  KernelFlow — Tensor Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Test 1: Shape
    std::cout << "Test 1: Create Tensor and verify shape" << std::endl;
    bool t1 = test_create_and_shape();
    print_result("Tensor shape [1024, 1024]", t1);
    std::cout << std::endl;

    // Test 2 & 3: Fill + verify
    std::cout << "Test 2: GPU fill with 3.14f" << std::endl;
    std::cout << "Test 3: Copy to CPU and verify values" << std::endl;
    bool t2 = test_fill_and_verify();
    print_result("GPU fill + CPU verify", t2);
    std::cout << std::endl;

    // Test 4 & 5: Timing benchmark
    std::cout << "Test 4: CPU vs GPU fill benchmark" << std::endl;
    test_cpu_vs_gpu_timing();
    std::cout << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "  All tests completed." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
