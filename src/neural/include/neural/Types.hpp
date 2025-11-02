#pragma once

#include <concepts>
#include <vector>

namespace neural_net
{

template <std::floating_point T>
using Tensor = std::vector<T>;

template <std::floating_point T>
using Matrix = std::vector<std::vector<T>>;

template <std::floating_point T>
struct TrainingData
{
    std::vector<Tensor<T>> inputs;
    std::vector<Tensor<T>> targets;
};

struct TrainingConfig
{
    double learning_rate = 0.01;
    int epochs = 10;
    int batch_size = 32;
    bool shuffle = true;
};

template <std::floating_point T>
struct TrainingResult
{
    T loss;
    T accuracy;
    int epoch;
};

} // namespace neural_net
