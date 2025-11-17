package neuralnet

import (
	"nn/internal/math"

	"golang.org/x/exp/constraints"
)

type Layer[T constraints.Float] interface {
	Forward(input *math.Matrix[T]) (*math.Matrix[T], error)
	Backward(gradient *math.Matrix[T], learningRate T) (*math.Matrix[T], error)
	Initialize(inputSize, outputSize int)
}
