package math_test

import (
	"nn/internal/math"
	"testing"
)

func TestMatrixNew(t *testing.T) {
	testcases := []struct {
		name  string
		input [2]int
		err   bool
	}{
		{"NewMatrix(0,0)", [2]int{0, 0}, true},
		{"NewMatrix(3,3)", [2]int{3, 3}, false},
		{"NewMatrix(-1,1)", [2]int{-1, 1}, true},
		{"NewMatrix(1,-1)", [2]int{1, -1}, true},
	}

	for _, tt := range testcases {
		t.Run(tt.name, func(t *testing.T) {
			m, err := math.NewMatrix[float64](tt.input[0], tt.input[1])

			if (err != nil) != tt.err {
				t.Errorf("TestMatrixNew() error = %v", err)
				return
			}

			if tt.err {
				if m != nil {
					t.Errorf("TestMatrixNew() matrix should be nil when error occurs, got %v", m)
				}
				return
			}

			if m.Rows() != tt.input[0] {
				t.Errorf("TestMatrixNew() rows = %v, expected %v", m.Rows(), tt.input[0])
			}
			if m.Cols() != tt.input[1] {
				t.Errorf("TestMatrixNew() cols = %v, expected %v", m.Cols(), tt.input[1])
			}
		})
	}
}

func TestMatrixAdd(t *testing.T) {
	t.Run("add two matrices of same size", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m1.Set(0, 0, 1.0)
		m1.Set(0, 1, 2.0)
		m1.Set(1, 0, 3.0)
		m1.Set(1, 1, 4.0)

		m2, _ := math.NewMatrix[float64](2, 2)
		m2.Set(0, 0, 5.0)
		m2.Set(0, 1, 6.0)
		m2.Set(1, 0, 7.0)
		m2.Set(1, 1, 8.0)

		result, err := m1.Add(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expected := [][]float64{
			{6.0, 8.0},
			{10.0, 12.0},
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, result.Get(i, j))
				}
			}
		}
	})

	t.Run("add matrices with negative values", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m1.Set(0, 0, -1.0)
		m1.Set(0, 1, 2.0)
		m1.Set(1, 0, 0.0)
		m1.Set(1, 1, -3.0)

		m2, _ := math.NewMatrix[float64](2, 2)
		m2.Set(0, 0, 4.0)
		m2.Set(0, 1, -2.0)
		m2.Set(1, 0, 1.0)
		m2.Set(1, 1, 5.0)

		result, err := m1.Add(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expected := [][]float64{
			{3.0, 0.0},
			{1.0, 2.0},
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, result.Get(i, j))
				}
			}
		}
	})

	t.Run("add matrices of different rows", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m2, _ := math.NewMatrix[float64](3, 2)

		_, err := m1.Add(m2)
		if err == nil {
			t.Error("Expected error for different row sizes, but got none")
		}
	})

	t.Run("add matrices of different cols", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m2, _ := math.NewMatrix[float64](2, 3)

		_, err := m1.Add(m2)
		if err == nil {
			t.Error("Expected error for different column sizes, but got none")
		}
	})

	t.Run("add identity matrices", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m1.Set(0, 0, 1.0)
		m1.Set(1, 1, 1.0)

		m2, _ := math.NewMatrix[float64](2, 2)
		m2.Set(0, 0, 1.0)
		m2.Set(1, 1, 1.0)

		result, err := m1.Add(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expected := [][]float64{
			{2.0, 0.0},
			{0.0, 2.0},
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, result.Get(i, j))
				}
			}
		}
	})

	t.Run("add zero matrices", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m2, _ := math.NewMatrix[float64](2, 2)

		result, err := m1.Add(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != 0.0 {
					t.Errorf("Expected 0 at (%d,%d), got %v", i, j, result.Get(i, j))
				}
			}
		}
	})
}

func TestAddFunc(t *testing.T) {
	t.Run("add identity matrices", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m1.Set(0, 0, 1.0)
		m1.Set(1, 1, 1.0)

		m2, _ := math.NewMatrix[float64](2, 2)
		m2.Set(0, 0, 1.0)
		m2.Set(1, 1, 1.0)

		result, err := m1.Add(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expected := [][]float64{
			{2.0, 0.0},
			{0.0, 2.0},
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, result.Get(i, j))
				}
			}
		}
	})
}

func TestMatrixMul(t *testing.T) {
	t.Run("multiply compatible matrices", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 3)
		m1.Set(0, 0, 1.0)
		m1.Set(0, 1, 2.0)
		m1.Set(0, 2, 3.0)
		m1.Set(1, 0, 4.0)
		m1.Set(1, 1, 5.0)
		m1.Set(1, 2, 6.0)

		m2, _ := math.NewMatrix[float64](3, 2)
		m2.Set(0, 0, 7.0)
		m2.Set(0, 1, 8.0)
		m2.Set(1, 0, 9.0)
		m2.Set(1, 1, 10.0)
		m2.Set(2, 0, 11.0)
		m2.Set(2, 1, 12.0)

		result, err := m1.Mul(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Expected result:
		// [1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58, 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64]
		// [4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139, 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154]
		expected := [][]float64{
			{58.0, 64.0},
			{139.0, 154.0},
		}

		if result.Rows() != 2 || result.Cols() != 2 {
			t.Errorf("Expected result size 2x2, got %dx%d", result.Rows(), result.Cols())
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, result.Get(i, j))
				}
			}
		}
	})

	t.Run("multiply incompatible matrices should fail", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 3)
		m2, _ := math.NewMatrix[float64](4, 2) // cols(3) != rows(4)

		_, err := m1.Mul(m2)
		if err == nil {
			t.Error("Expected error for incompatible matrices, but got none")
		}
	})

	t.Run("multiply identity matrix", func(t *testing.T) {
		identity, _ := math.NewMatrix[float64](2, 2)
		identity.Set(0, 0, 1.0)
		identity.Set(1, 1, 1.0)

		m, _ := math.NewMatrix[float64](2, 2)
		m.Set(0, 0, 5.0)
		m.Set(0, 1, 6.0)
		m.Set(1, 0, 7.0)
		m.Set(1, 1, 8.0)

		result, err := m.Mul(identity)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Multiplying by identity should return the original matrix
		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != m.Get(i, j) {
					t.Errorf("Expected %v at (%d,%d), got %v", m.Get(i, j), i, j, result.Get(i, j))
				}
			}
		}
	})

	t.Run("multiply by zero matrix", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 3)
		m1.Set(0, 0, 1.0)
		m1.Set(0, 1, 2.0)
		m1.Set(1, 0, 3.0)
		m1.Set(1, 1, 4.0)

		zero, _ := math.NewMatrix[float64](3, 2)

		result, err := m1.Mul(zero)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Result should be zero matrix
		for i := 0; i < result.Rows(); i++ {
			for j := 0; j < result.Cols(); j++ {
				if result.Get(i, j) != 0.0 {
					t.Errorf("Expected 0 at (%d,%d), got %v", i, j, result.Get(i, j))
				}
			}
		}
	})

	t.Run("multiply 1x1 matrices", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](1, 1)
		m1.Set(0, 0, 5.0)

		m2, _ := math.NewMatrix[float64](1, 1)
		m2.Set(0, 0, 3.0)

		result, err := m1.Mul(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if result.Get(0, 0) != 15.0 {
			t.Errorf("Expected 15, got %v", result.Get(0, 0))
		}
	})

	t.Run("multiply rectangular matrices", func(t *testing.T) {
		// 1x3 * 3x1 = 1x1
		m1, _ := math.NewMatrix[float64](1, 3)
		m1.Set(0, 0, 1.0)
		m1.Set(0, 1, 2.0)
		m1.Set(0, 2, 3.0)

		m2, _ := math.NewMatrix[float64](3, 1)
		m2.Set(0, 0, 4.0)
		m2.Set(1, 0, 5.0)
		m2.Set(2, 0, 6.0)

		result, err := m1.Mul(m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expected := 1.0*4.0 + 2.0*5.0 + 3.0*6.0 // = 4 + 10 + 18 = 32
		if result.Get(0, 0) != expected {
			t.Errorf("Expected %v, got %v", expected, result.Get(0, 0))
		}
		if result.Rows() != 1 || result.Cols() != 1 {
			t.Errorf("Expected result size 1x1, got %dx%d", result.Rows(), result.Cols())
		}
	})
}

func TestMulFunc(t *testing.T) {
	t.Run("test standalone Multiply function", func(t *testing.T) {
		m1, _ := math.NewMatrix[float64](2, 2)
		m1.Set(0, 0, 1.0)
		m1.Set(0, 1, 2.0)
		m1.Set(1, 0, 3.0)
		m1.Set(1, 1, 4.0)

		m2, _ := math.NewMatrix[float64](2, 2)
		m2.Set(0, 0, 2.0)
		m2.Set(0, 1, 0.0)
		m2.Set(1, 0, 1.0)
		m2.Set(1, 1, 2.0)

		result, err := math.Mul(m1, m2)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		expected := [][]float64{
			{4.0, 4.0},  // 1*2 + 2*1 = 4, 1*0 + 2*2 = 4
			{10.0, 8.0}, // 3*2 + 4*1 = 10, 3*0 + 4*2 = 8
		}

		for i := range 2 {
			for j := range 2 {
				if result.Get(i, j) != expected[i][j] {
					t.Errorf("Expected %v at (%d,%d), got %v", expected[i][j], i, j, result.Get(i, j))
				}
			}
		}
	})
}
