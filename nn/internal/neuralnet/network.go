package neuralnet

import "golang.org/x/exp/constraints"

type Network[T constraints.Float] struct {
	layers []Layer[T]
}
