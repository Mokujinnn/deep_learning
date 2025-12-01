package neuralnet

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type Danse[T constraints.Float] struct {
	inputSize  int
	outputSize int
	weights    [][]T
	biases     []T
	lastInput  []T
	lastOutput []T
}

func NewDanse[T constraints.Float](outputSize int) *Danse[T] {
	return &Danse[T]{
		outputSize: outputSize,
	}
}

func (l *Danse[T]) Forward(input []T) ([]T, error) {
	if len(input) != l.inputSize {
		return nil, fmt.Errorf("input shape must have size = %v, but have size = %v", l.inputSize, len(input))
	}

	copy(l.lastInput, input)

	for i := range l.outputSize {
		sum := l.biases[i]

		for j := range l.inputSize {
			sum += input[j] * l.weights[i][j]
		}

		l.lastOutput[i] = sum
	}

	return l.lastOutput, nil
}

func (l *Danse[T]) Backward(gradient []T, learningRate float64) ([]T, error) {
	if len(gradient) != l.outputSize {
		return nil, fmt.Errorf("gradient shape must have size = %v, but have size = %v", l.outputSize, len(gradient))
	}

	inputGradient := make([]T, l.inputSize)

	for i := range l.outputSize {
		for j := range l.inputSize {
			inputGradient[j] += gradient[i] * l.weights[i][j]
			l.weights[i][j] -= T(learningRate) * gradient[i] * l.lastInput[j]
		}

		l.biases[i] -= T(learningRate) * gradient[i]
	}

	return inputGradient, nil
}

func (l *Danse[T]) Initialize(inputSize int) {
	l.inputSize = inputSize
	l.weights = make([][]T, l.outputSize)
	for i := range l.weights {
		l.weights[i] = make([]T, l.inputSize)
	}
	l.biases = make([]T, l.outputSize)

	for i := range l.outputSize {
		for j := range l.inputSize {
			l.weights[i][j] = 1
		}
		l.biases[i] = 0.1
	}
}

func (l *Danse[T]) GetOutputSize() int {
	return l.outputSize
}

func (l *Danse[T]) GetWeights() []T {
	// TODO: Implement
	return []T{}
}

func (l *Danse[T]) GetBiases() []T {
	// TODO: Implement
	return []T{}
}

func (l *Danse[T]) SetWeights(weights []T) {
	// TODO: Implement
}

func (l *Danse[T]) SetBiases(biases []T) {
	// TODO: Implement
}

func (l *Danse[T]) GetName() string {
	return "Danse"
}
