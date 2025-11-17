package neuralnet

import (
	"nn/internal/nnmath"

	"golang.org/x/exp/constraints"
)

type Layer[T constraints.Float] interface {
	Forward(input *nnmath.Matrix[T]) (*nnmath.Matrix[T], error)
	Backward(gradient *nnmath.Matrix[T], learningRate T) (*nnmath.Matrix[T], error)
	Initialize(inputSize, outputSize int)
}
