#include "NeuralNetwork.hpp"

#include "Danse.hpp"

namespace neural_net
{

NeuralNetwork::NeuralNetwork()
{
    layers::Danse<float> d(64);
}

} // namespace neural_net
