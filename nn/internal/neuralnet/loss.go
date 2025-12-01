package neuralnet

type Loss interface {
	Compute(prediction []float64, target []float64) float64
	Derevative(prediction []float64, target []float64) []float64
}
