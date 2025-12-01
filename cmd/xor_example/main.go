package main

import (
	"fmt"
	"math"

	"nn/internal/neuralnet"
)

func main() {

	fmt.Println("=== Нейросеть для XOR ===")

	X_train := [][]float64{
		{0, 0}, // 0 XOR 0 = 0
		{0, 1}, // 0 XOR 1 = 1
		{1, 0}, // 1 XOR 0 = 1
		{1, 1}, // 1 XOR 1 = 0
	}

	Y_train := [][]float64{
		{0}, // Результат для [0, 0]
		{1}, // Результат для [0, 1]
		{1}, // Результат для [1, 0]
		{0}, // Результат для [1, 1]
	}

	network := CreateXORNetwork()

	fmt.Println("\nНачинаем обучение...")
	network.Fit(X_train, Y_train, 0.1, 100000, false)

	fmt.Println("\n=== Тестирование ===")
	TestNetwork(network, X_train, Y_train)

	fmt.Println("\n=== Дополнительные тесты ===")
	TestRandomValues(network)
}

func CreateXORNetwork() *neuralnet.Network[float64] {
	network := neuralnet.NewNetwork[float64]()

	network.AddLayer(neuralnet.NewDanse[float64](10))
	network.AddLayer(neuralnet.NewActivationLayer[float64](neuralnet.Sigmoid))
	network.AddLayer(neuralnet.NewDanse[float64](1))
	network.AddLayer(neuralnet.NewActivationLayer[float64](neuralnet.Sigmoid))

	network.Compile(2, neuralnet.NewMSELoss[float64]())

	return network
}

func TestNetwork(network *neuralnet.Network[float64], X [][]float64, Y [][]float64) {
	fmt.Println("Вход -> Ожидаемый -> Предсказанный -> Ошибка")
	fmt.Println("---------------------------------------------")

	for i := 0; i < len(X); i++ {
		prediction, err := network.Predict(X[i])
		if err != nil {
			fmt.Printf("Ошибка предсказания: %v\n", err)
			continue
		}

		expected := Y[i][0]
		pred := prediction[0]
		error := math.Abs(expected - pred)

		fmt.Printf("[%.0f, %.0f] -> %.1f -> %.4f -> %.4f\n",
			X[i][0], X[i][1], expected, pred, error)
	}
}

func TestRandomValues(network *neuralnet.Network[float64]) {
	fmt.Println("Случайные значения (округленные до 0 или 1):")

	testCases := [][]float64{
		{0.1, 0.9},  // ≈ [0, 1] -> 1
		{0.8, 0.3},  // ≈ [1, 0] -> 1
		{0.9, 0.85}, // ≈ [1, 1] -> 0
		{0.05, 0.1}, // ≈ [0, 0] -> 0
		{0.6, 0.6},  // ≈ [1, 1] -> 0
		{0.4, 0.7},  // ≈ [0, 1] -> 1
	}

	for _, test := range testCases {
		prediction, err := network.Predict(test)
		if err != nil {
			fmt.Printf("Ошибка: %v\n", err)
			continue
		}

		a := 0
		if test[0] > 0.5 {
			a = 1
		}
		b := 0
		if test[1] > 0.5 {
			b = 1
		}

		expected := a ^ b
		pred := 0
		if prediction[0] > 0.5 {
			pred = 1
		}

		fmt.Printf("[%.1f, %.1f] ≈ [%d, %d]: XOR = %d, Предсказано = %d (%.4f) %s\n",
			test[0], test[1], a, b, expected, pred, prediction[0],
			checkMark(expected, pred))
	}
}

func checkMark(expected, pred int) string {
	if expected == pred {
		return "✓"
	}
	return "✗"
}
