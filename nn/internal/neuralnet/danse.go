package neuralnet

import (
	"fmt"
	"nn/internal/math"

	"golang.org/x/exp/constraints"
)

type Danse[T constraints.Float] struct {
	inputSize  int
	outputSize int
	weights    *math.Matrix[T]
	biases     *math.Matrix[T]
	lastInput  *math.Matrix[T]
	lastOutput *math.Matrix[T]
}

func NewDanse[T constraints.Float](outputSize int) *Danse[T] {
	return &Danse[T]{
		outputSize: outputSize,
	}
}

func (l *Danse[T]) Forward(input *math.Matrix[T]) (*math.Matrix[T], error) {
	if input.Rows() != 1 || input.Cols() != l.inputSize {
		return nil, fmt.Errorf("Input shape must be: rows = %v, cols = %v, but have size: rows = %v, cols = %v", 1, l.inputSize, input.Rows(), input.Cols())
	}

	l.lastInput = input.Copy()

	for i := range l.outputSize {
		sum := l.biases.Get(0, i)

		for j := range l.inputSize {
			sum += input.Get(0, i) * l.weights.Get(i, j)
		}

		l.lastOutput.Set(0, i, sum)
	}

	return l.lastOutput, nil
}

func (l *Danse[T]) Backward(gradient *math.Matrix[T], learningRate T) (*math.Matrix[T], error) {
	if gradient.Rows() != 1 || gradient.Cols() != l.outputSize {
		return nil, fmt.Errorf("Gradient shape must be: rows = %v, cols = %v, but have size: rows = %v, cols = %v", 1, l.outputSize, gradient.Rows(), gradient.Cols())
	}

	inputGradient, _ := math.NewMatrix[T](1, l.inputSize)

	for i := range l.outputSize {
		for j := range l.inputSize {
			inputGradient.Set(0, i, inputGradient.Get(0, i)+gradient.Get(0, i)*l.weights.Get(i, j))
			l.weights.Set(i, j, l.weights.Get(i, j)-learningRate*gradient.Get(0, i)*l.lastInput.Get(0, j))
		}

		l.biases.Set(0, i, l.biases.Get(0, i)-learningRate+gradient.Get(0, i))
	}

	return inputGradient, nil
}

func (l *Danse[T]) Initialize(inputSize int) {
	l.inputSize = inputSize
	l.weights, _ = math.NewMatrix[T](l.outputSize, l.inputSize)
	l.biases, _ = math.NewMatrix[T](1, l.outputSize)

	for i := range l.outputSize {
		for j := range l.inputSize {
			l.weights.Set(i, j, 1)
		}
		l.biases.Set(0, i, 0.1)
	}
}
