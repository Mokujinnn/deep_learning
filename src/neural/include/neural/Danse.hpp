#pragma once

#include <ranges>
#include <stdexcept>

#include "Layer.hpp"

namespace neural_net
{

namespace layers
{

template <std::floating_point T>
class Danse : public Layer<T>
{
private:
    int input_size_;
    int output_size_;
    Matrix<T> weights_;
    Tensor<T> biases_;
    Tensor<T> last_input_;
    Tensor<T> last_output_;

public:
    Danse(int output_size);

    void initialize(int input_size) override;
    Tensor<T> forward(const Tensor<T> &input) override;
    Tensor<T> backward(const Tensor<T> &gradient, T learning_rate) override;
    int get_output_size() const override;
    std::string get_name() const override;

    Matrix<T> get_weights() const override;
    Tensor<T> get_biases() const override;
    void set_weights(const Matrix<T> &weights) override;
    void set_biases(const Tensor<T> &biases) override;
};

template <std::floating_point T>
inline Danse<T>::Danse(int output_size)
    : output_size_(output_size)
    , last_output_(output_size)
{
}

template <std::floating_point T>
inline void Danse<T>::initialize(int input_size)
{
}

template <std::floating_point T>
inline Tensor<T> Danse<T>::forward(const Tensor<T> &input)
{
    if (input.size() != input_size_)
    {
        throw std::invalid_argument("Input size mismatch");
    }

    last_input_ = input;

    for (int i : std::views::iota(0, output_size_))
    {
        T sum = biases_[i];

        for (int j : std::views::iota(0, input_size_))
        {
            sum += input[i] * weights_[i][j];
        }

        last_output_[i] = sum;
    }

    return last_output_;
}

template <std::floating_point T>
inline Tensor<T> Danse<T>::backward(const Tensor<T> &gradient, T learning_rate)
{
    if (gradient.size() != output_size_)
    {
        throw std::invalid_argument("Gradient size mismatch");
    }

    Tensor<T> input_gradient(input_size_, static_cast<T>(0.0));

    for (int i : std::views::iota(0, output_size_))
    {
        for (int j : std::views::iota(0, input_size_))
        {
            input_gradient[i] += gradient[i] * weights_[i][j];
            weights_[i][j] -= learning_rate * gradient[i] * last_input_[j];
        }

        biases_[i] += learning_rate * gradient[i];
    }

    return input_gradient;
}

template <std::floating_point T>
inline int Danse<T>::get_output_size() const
{
    return output_size_;
}

template <std::floating_point T>
inline std::string Danse<T>::get_name() const
{
    return "Danse";
}

template <std::floating_point T>
inline Matrix<T> Danse<T>::get_weights() const
{
    return weights_;
}

template <std::floating_point T>
inline Tensor<T> Danse<T>::get_biases() const
{
    return biases_;
}

} // namespace layers

} // namespace neural_net
