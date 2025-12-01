package neuralnet

import (
	"math"

	"golang.org/x/exp/constraints"
)

type MSELoss[T constraints.Float] struct{}

func NewMSELoss[T constraints.Float]() *MSELoss[T] {
	return &MSELoss[T]{}
}

func (l *MSELoss[T]) Compute(prediction []T, target []T) T {
	if len(prediction) != len(target) {
		panic("Prediction and target must have the same size")
	}

	sum := T(0.0)
	for i := range prediction {
		sum += T(math.Pow(float64(prediction[i]-target[i]), 2))
	}

	return sum / T(len(prediction))
}

func (l *MSELoss[T]) Derevative(prediction []T, target []T) []T {
	if len(prediction) != len(target) {
		panic("Prediction and target must have the same size")
	}

	result := make([]T, len(prediction))
	for i := range prediction {
		result[i] = prediction[i] - target[i]
	}

	return result
}
