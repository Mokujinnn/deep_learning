#pragma once

#include <string>
#include <vector>

namespace neural_net
{

template <typename T>
class Layer
{
private:
public:
    ~Layer() = default;

    virtual void initialize(int input_size) = 0;

    virtual std::vector<T> forward(const std::vector<T>)
};

} // namespace neural_net
