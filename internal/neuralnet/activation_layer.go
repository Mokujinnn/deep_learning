package neuralnet

import (
	"fmt"
	"math"

	"golang.org/x/exp/constraints"
)

type ActivationFuncType int

const (
	Relu ActivationFuncType = iota
	Sigmoid
	Tanh
)

type ActivationLayer[T constraints.Float] struct {
	size       int
	lastInput  []T
	lastOutput []T
	f          func(x float64) float64
	fd         func(x float64) float64
	name       string
}

func NewActivationLayer[T constraints.Float](t ActivationFuncType) *ActivationLayer[T] {
	result := &ActivationLayer[T]{}

	switch t {
	case Relu:
		result.f = relu
		result.fd = reluDerevative
		result.name = "Activation Relu"
	case Sigmoid:
		result.f = sigmoid
		result.fd = sigmoidDerivative
		result.name = "Activation Sigmoid"
	case Tanh:
		result.f = tanh
		result.fd = tanhDerivative
		result.name = "Activation Tanh"
	}

	return result
}

func (l *ActivationLayer[T]) Forward(input []T) ([]T, error) {
	if len(input) != l.size {
		return nil, fmt.Errorf("input shape must have size = %v, but have size = %v", l.size, len(input))
	}

	copy(l.lastInput, input)

	for i := range l.size {
		l.lastOutput[i] = T(l.f(float64(input[i])))
	}

	return l.lastOutput, nil
}

func (l *ActivationLayer[T]) Backward(gradient []T, learningRate float64) ([]T, error) {
	if len(gradient) != l.size {
		return nil, fmt.Errorf("gradient shape must have size = %v, but have size = %v", l.size, len(gradient))
	}

	inputGradient := make([]T, l.size)

	for i := range l.size {
		inputGradient[i] = gradient[i] * T(l.fd(float64(l.lastInput[i])))
	}

	return inputGradient, nil
}

func (l *ActivationLayer[T]) Initialize(inputSize int) {
	l.size = inputSize
	l.lastInput = make([]T, inputSize)
	l.lastOutput = make([]T, inputSize)
}

func (l *ActivationLayer[T]) GetOutputSize() int {
	return l.size
}

func (l *ActivationLayer[T]) GetWeights() []T {
	// TODO: Implement
	return []T{}
}

func (l *ActivationLayer[T]) GetBiases() []T {
	// TODO: Implement
	return []T{}
}

func (l *ActivationLayer[T]) SetWeights(weights []T) {
	// TODO: Implement
}

func (l *ActivationLayer[T]) SetBiases(biases []T) {
	// TODO: Implement
}

func (l *ActivationLayer[T]) GetName() string {
	return l.name
}

func relu(x float64) float64 {
	return math.Max(0.0, x)
}

func reluDerevative(x float64) float64 {
	result := 0.0
	if x > 0 {
		result = 1.0
	}

	return result
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func tanhDerivative(x float64) float64 {
	return 1.0 - x*x
}
