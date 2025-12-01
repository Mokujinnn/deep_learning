package neuralnet

import (
	"golang.org/x/exp/constraints"
)

type Layer[T constraints.Float] interface {
	Forward(input []T) ([]T, error)
	Backward(gradient []T, learningRate float64) ([]T, error)
	Initialize(inputSize int)
	GetOutputSize() int
	GetWeights() []T
	GetBiases() []T
	SetWeights(weights []T)
	SetBiases(biases []T)
	GetName() string
}
