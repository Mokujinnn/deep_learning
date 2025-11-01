#pragma once

#include "Layer.hpp"

namespace neural_net
{

namespace layers
{

template <typename T>
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
    Tensor<T> backwadr(const Tensor<T> &gradient, T learning_rate) override;
    int get_output_size() const override;
    std::string get_name() const override;

    Matrix<T> get_weights() const override;
    Tensor<T> get_biases() const override;
    void set_weights(const Matrix<T> &weights) override;
    void set_biases(const Tensor<T> &biases) override;
};

template <typename T>
inline Danse<T>::Danse(int output_size)
    : output_size(output_size_)
{
}

template <typename T>
inline int Danse<T>::get_output_size() const
{
    return output_size_;
}

template <typename T>
inline std::string Danse<T>::get_name() const
{
    return "Danse";
}

template <typename T>
inline Matrix<T> Danse<T>::get_weights() const
{
    return weights_;
}

template <typename T>
inline Tensor<T> Danse<T>::get_biases() const
{
    return biases_;
}

} // namespace layers

} // namespace neural_net
