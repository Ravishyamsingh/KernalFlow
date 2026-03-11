#pragma once

#include "tensor.hpp"
#include "kernels.hpp"
#include "layers.hpp"
#include <vector>
#include <string>

// ============================================================================
// Layer — Abstract Base Class
// ============================================================================
//
// All layers in KernelFlow inherit from this interface.
// Enables polymorphic storage in Sequential container.
//
// Each layer must implement:
//   forward()   — compute output from input (on GPU)
//   name()      — return human-readable layer name (for print_summary)
//
// ============================================================================
class Layer {
public:
    virtual Tensor forward(Tensor& input) = 0;
    virtual std::string name() const = 0;
    virtual ~Layer() = default;
};

// ============================================================================
// ReLULayer — Wraps launch_relu kernel
// ============================================================================
//
// In-place activation: f(x) = max(0, x)
// Does NOT allocate new memory — modifies input directly and returns a
// move of the input to maintain ownership semantics.
//
// Note: This modifies the input tensor in-place for efficiency.
// ============================================================================
class ReLULayer : public Layer {
public:
    Tensor forward(Tensor& input) override;
    std::string name() const override { return "ReLU"; }
};

// ============================================================================
// SoftmaxLayer — Wraps launch_softmax kernel
// ============================================================================
//
// Per-row softmax: converts logits → probability distribution.
// Expects input shape [batch, features].
// In-place operation like ReLU.
// ============================================================================
class SoftmaxLayer : public Layer {
public:
    Tensor forward(Tensor& input) override;
    std::string name() const override { return "Softmax"; }
};

// ============================================================================
// LinearLayerAdapter — Wraps LinearLayer to fit the Layer interface
// ============================================================================
//
// LinearLayer was written before the Layer base class existed.
// This adapter owns a LinearLayer and delegates forward() to it.
// ============================================================================
class LinearLayerAdapter : public Layer {
private:
    LinearLayer* linear_;
    bool owns_;

public:
    LinearLayerAdapter(int in_features, int out_features);
    ~LinearLayerAdapter() override;

    Tensor forward(Tensor& input) override;
    std::string name() const override;

    // Access underlying LinearLayer (for load_weights, etc.)
    LinearLayer* get() { return linear_; }
};

// ============================================================================
// Sequential — Model Container
// ============================================================================
//
// Chains multiple layers in sequence. Data flows:
//   input → layer[0] → layer[1] → ... → layer[n-1] → output
//
// Usage:
//   Sequential model;
//   model.add(new LinearLayerAdapter(4, 128));
//   model.add(new ReLULayer());
//   model.add(new LinearLayerAdapter(128, 2));
//   model.add(new SoftmaxLayer());
//
//   Tensor output = model.forward(input);
//
// ============================================================================
class Sequential : public Layer {
private:
    std::vector<Layer*> layers_;

public:
    Sequential() = default;

    // Add a layer — Sequential takes ownership
    void add(Layer* layer);

    // Forward pass: chains all layers sequentially
    Tensor forward(Tensor& input) override;

    // Print model architecture summary
    void print_summary() const;

    std::string name() const override { return "Sequential"; }

    // Get number of layers
    int num_layers() const { return static_cast<int>(layers_.size()); }

    // Access individual layer
    Layer* get_layer(int index) { return layers_[index]; }

    // Destructor: deletes all owned layers
    ~Sequential() override;
};