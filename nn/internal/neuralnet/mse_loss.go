package neuralnet

import "math"

type MSELoss struct{}

func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

func (l *MSELoss) Compute(prediction []float64, target []float64) float64 {
	if len(prediction) != len(target) {
		panic("Prediction and target must have the same size")
	}

	sum := 0.0
	for i := range prediction {
		sum += math.Pow(prediction[i]-target[i], 2)
	}

	return sum / float64(len(prediction))
}

func (l *MSELoss) Derevative(prediction []float64, target []float64) []float64 {
	if len(prediction) != len(target) {
		panic("Prediction and target must have the same size")
	}

	result := make([]float64, len(prediction))
	for i := range prediction {
		result[i] = prediction[i] - target[i]
	}

	return result
}
