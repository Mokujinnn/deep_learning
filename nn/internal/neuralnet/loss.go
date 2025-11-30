package neuralnet

import (
	"golang.org/x/exp/constraints"
)

type Loss[T constraints.Float] interface {
	Compute(prediction []T, target []T) T
	Gradient(prediction []T, target []T) []T
}
