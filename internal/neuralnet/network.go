package neuralnet

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type Network[T constraints.Float] struct {
	layers []Layer[T]
	loss   Loss[T]
}

func NewNetwork[T constraints.Float]() *Network[T] {
	return &Network[T]{
		layers: make([]Layer[T], 0),
	}
}

func (n *Network[T]) AddLayer(layer Layer[T]) {
	n.layers = append(n.layers, layer)
}

func (n *Network[T]) Compile(inputSize int, loss Loss[T]) {
	n.initialize(inputSize)
	n.loss = loss
}

func (n *Network[T]) Fit(X [][]T, Y [][]T, learningRate float64, epochs int, verbose bool) {
	if len(X) != len(Y) {
		panic("X and Y must have the same size")
	}

	for epoch := range epochs {
		if verbose {
			fmt.Printf("Epoch: %v in %v\n", epoch, epochs)
		}

		for i := range X {
			prediction, err := n.forward(X[i])
			if err != nil {
				panic(err)
			}

			err = n.backward(prediction, Y[i], learningRate)
			if err != nil {
				panic(err)
			}
		}
	}
}

func (n *Network[T]) Predict(input []T) ([]T, error) {
	lastInput := input

	var err error

	for _, layer := range n.layers {
		lastInput, err = layer.Forward(lastInput)
		if err != nil {
			return nil, err
		}
	}

	return lastInput, nil
}

func (n *Network[T]) forward(input []T) ([]T, error) {
	var err error

	lastInput := input
	for _, layer := range n.layers {
		lastInput, err = layer.Forward(lastInput)
		if err != nil {
			return nil, err
		}
	}

	return lastInput, nil
}

func (n *Network[T]) backward(prediction []T, target []T, learningRate float64) error {
	gradient := n.loss.Derevative(prediction, target)

	var err error

	for i := len(n.layers) - 1; i >= 0; i-- {
		gradient, err = n.layers[i].Backward(gradient, learningRate)
		if err != nil {
			return err
		}
	}

	return nil
}

func (n *Network[T]) initialize(inputSize int) {
	curentSize := inputSize
	for _, layer := range n.layers {
		layer.Initialize(curentSize)
		curentSize = layer.GetOutputSize()
	}
}
