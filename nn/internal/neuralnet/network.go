package neuralnet

import (
	"golang.org/x/exp/constraints"
)

type Network[T constraints.Float] struct {
	layers []Layer[T]
	loss   Loss
}

func NewNetwork[T constraints.Float]() *Network[T] {
	return &Network[T]{
		layers: make([]Layer[T], 0),
	}
}

func (n *Network[T]) AddLayer(layer Layer[T]) {
	n.layers = append(n.layers, layer)
}

func (n *Network[T]) Compile(inputSize int, loss Loss) {
	n.initialize(inputSize)
	n.loss = loss
}

func (n *Network[T]) initialize(inputSize int) {
	curentSize := inputSize
	for _, layer := range n.layers {
		layer.Initialize(curentSize)
		curentSize = layer.GetOutputSize()
	}
}
