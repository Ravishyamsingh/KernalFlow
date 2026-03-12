// ============================================================================
// KernelFlow — Sequential Model + Layer Implementations
// ============================================================================

#include "model.hpp"
#include <cstdio>

// ============================================================================
// ReLULayer Implementation
// ============================================================================
//
// Applies ReLU in-place on the GPU data, then moves the tensor out.
// The caller receives ownership of the (now-modified) tensor.
// ============================================================================
Tensor ReLULayer::forward(Tensor& input) {
    launch_relu(input.gpu_ptr(), input.size());
    return std::move(input);
}


// ============================================================================
// SoftmaxLayer Implementation
// ============================================================================
//
// Applies per-row softmax in-place. Expects shape [batch, features].
// ============================================================================
Tensor SoftmaxLayer::forward(Tensor& input) {
    auto shape = input.get_shape();
    int batch = shape[0];
    int features = shape[1];
    launch_softmax(input.gpu_ptr(), features, batch);
    return std::move(input);
}

// ============================================================================
// LinearLayerAdapter Implementation
// ============================================================================

LinearLayerAdapter::LinearLayerAdapter(int in_features, int out_features)
    : linear_(new LinearLayer(in_features, out_features)), owns_(true) {}

LinearLayerAdapter::~LinearLayerAdapter() {
    if (owns_) delete linear_;
}

Tensor LinearLayerAdapter::forward(Tensor& input) {
    return linear_->forward(input);
}

std::string LinearLayerAdapter::name() const {
    return "Linear(" + std::to_string(linear_->get_in_features()) +
           " -> " + std::to_string(linear_->get_out_features()) + ")";
}

// ============================================================================
// Sequential Implementation
// ============================================================================

// Add layer — Sequential takes ownership
void Sequential::add(Layer* layer) {
    layers_.push_back(layer);
}

// Forward pass: chain all layers
//
// Data flows through each layer sequentially. Each forward() returns
// a new Tensor (or moves the input), which becomes the input for the
// next layer. The final output is returned to the caller.
Tensor Sequential::forward(Tensor& input) {
    // First layer takes the original input reference
    Tensor current = layers_[0]->forward(input);

    // Subsequent layers take the output of the previous layer
    for (int i = 1; i < static_cast<int>(layers_.size()); ++i) {
        current = layers_[i]->forward(current);
    }

    return current;
}

// Print model summary — layer index, type, and name
void Sequential::print_summary() const {
    printf("==============================================================\n");
    printf("  KernelFlow Model Summary\n");
    printf("==============================================================\n");
    printf("  %-6s  %-35s\n", "Layer", "Type");
    printf("--------------------------------------------------------------\n");

    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        printf("  %-6d  %-35s\n", i, layers_[i]->name().c_str());
    }

    printf("==============================================================\n");
    printf("  Total layers: %d\n", static_cast<int>(layers_.size()));
    printf("==============================================================\n");
}

// Destructor: delete all owned layers
Sequential::~Sequential() {
    for (auto* layer : layers_) {
        delete layer;
    }
    layers_.clear();
}
