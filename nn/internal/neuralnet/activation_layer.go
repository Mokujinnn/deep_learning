package neuralnet

import (
	"math"

	"nn/internal/nnmath"

	"golang.org/x/exp/constraints"
)

type ActivationFuncType int

const (
	Relu ActivationFuncType = iota
	Sigmoid
	Tanh
)

type ActivationLayer[T constraints.Float] struct {
	size   int
	input  *nnmath.Matrix[T]
	output *nnmath.Matrix[T]
	f      func(x float64) float64
}

func NewActivationLayer[T constraints.Float](t ActivationFuncType) *ActivationLayer[T] {
	result := &ActivationLayer[T]{}

	switch t {
	case Relu:
		result.f = relu
	case Sigmoid:
		result.f = sigmoid
	case Tanh:
		result.f = tanh
	}

	return result
}

func relu(x float64) float64 {
	return math.Max(0.0, x)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}
