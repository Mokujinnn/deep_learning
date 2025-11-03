#pragma once

#include <string>

#include "Types.hpp"

namespace neural_net
{

namespace layers
{

template <std::floating_point T>
class Layer
{
public:
    ~Layer() = default;

    virtual void initialize(int input_size) = 0;

    virtual Tensor<T> forward(const Tensor<T> &input) = 0;

    virtual Tensor<T> backward(const Tensor<T> &gradient, T learning_rate) = 0;

    virtual int get_output_size() const = 0;

    virtual std::string get_name() const = 0;

    virtual Matrix<T> get_weights() const
    {
        return {};
    }

    virtual Tensor<T> get_biases() const
    {
        return {};
    }

    virtual void set_weights(const Matrix<T> &weights)
    {
    }

    virtual void set_biases(const Tensor<T> &biases)
    {
    }
};

} // namespace layers

} // namespace neural_net
