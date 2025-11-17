package math

import (
	"fmt"

	"golang.org/x/exp/constraints"
)

type Matrix[T constraints.Float] struct {
	rows int
	cols int
	data []T
}

func NewMatrix[T constraints.Float](rows, cols int) (*Matrix[T], error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("Size must be positive: rows = %v cols = %v", rows, cols)
	}

	data := make([]T, rows*cols)

	return &Matrix[T]{
		rows: rows,
		cols: cols,
		data: data,
	}, nil
}

func (m *Matrix[T]) Rows() int {
	return m.rows
}

func (m *Matrix[T]) Cols() int {
	return m.cols
}

func (m *Matrix[T]) Get(row, col int) T {
	return m.data[row*m.cols+col]
}

func (m *Matrix[T]) Set(row, col int, value T) {
	m.data[row*m.cols+col] = value
}

func (m *Matrix[T]) Add(other *Matrix[T]) (*Matrix[T], error) {
	if m.rows != other.rows || m.cols != other.cols {
		return nil, fmt.Errorf("Matrices must have same size")
	}

	result, _ := NewMatrix[T](m.rows, m.cols)

	for i := range m.rows {
		for j := range m.cols {
			result.Set(i, j, m.Get(i, j)+other.Get(i, j))
		}
	}

	return result, nil
}

func Add[T constraints.Float](l, r *Matrix[T]) (*Matrix[T], error) {
	return l.Add(r)
}

func (m *Matrix[T]) Mul(other *Matrix[T]) (*Matrix[T], error) {
	if m.cols != other.rows {
		return nil, fmt.Errorf("Matrix1 cols != Matrix2 rows, m1 cols = %v, m2 rows = %v", m.cols, other.rows)
	}

	result, _ := NewMatrix[T](m.rows, other.cols)

	for i := range m.rows {
		for k := range m.cols {
			aik := m.Get(i, k)
			for j := range other.cols {
				cur := result.Get(i, j)
				result.Set(i, j, cur+aik*other.Get(k, j))
			}
		}
	}

	return result, nil
}

func Mul[T constraints.Float](l, r *Matrix[T]) (*Matrix[T], error) {
	return l.Mul(r)
}
