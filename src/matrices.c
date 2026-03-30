#include "matrices.h"
#include <stdio.h>
#include <stdlib.h>

struct Matrix { 
    double* values;
    int columns;
    int rows;
};

int _get_index(Matrix* matrix, int row, int column) {
    return (row * matrix->columns) + column;
}

void _set_matrix_value(Matrix* matrix, int row, int column, double value) {
    int index = _get_index(matrix, row, column);
    matrix->values[index] = value;
}

Matrix* create_matrix(int rows, int columns) {
    Matrix* curr_matrix = malloc(sizeof(Matrix));
    curr_matrix->values = malloc(sizeof(double) * rows * columns);
    curr_matrix->rows = rows;
    curr_matrix->columns = columns;
    return curr_matrix;
}

Matrix* create_matrix_with_values(int rows, int columns, double* values) {
    Matrix* allocated_matrix = create_matrix(rows, columns);
    for (int i = 0; i < rows*columns; i++) {
        allocated_matrix->values[i] = values[i];
    }
    return allocated_matrix;
}

void free_matrix(Matrix* matrix) {
    free(matrix->values);
    free(matrix);
}

void free_matrices(Matrix** matrices, int num_matrices) {
    for (int i = 0; i < num_matrices; i++) {
        free_matrix(matrices[i]);
    }
    free(matrices);
}

int set_matrix_value(Matrix* matrix, int row, int column, double value) {
    if (row >= matrix->rows || column >= matrix->columns) {
        fprintf(stderr, "Cannot set value, dimensions (%d, %d) out of bounds for m=(%d, %d)\n",
            row, column, matrix->rows, matrix->columns);
        exit(EXIT_FAILURE);
    }
    double* values = matrix->values;
    _set_matrix_value(matrix, row, column, value);
    return 1;
}

int get_matrix_value(Matrix* matrix, int row, int column) {
    int index = _get_index(matrix, row, column);
    return matrix->values[index];
}

int set_matrix_values(Matrix* matrix, int row, double* values, int length) {
    if (row >= matrix->rows || length > matrix->columns) {
        fprintf(stderr, "Cannot set values, dimensions out of bounds\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        _set_matrix_value(matrix, row, i, values[i]);
    }
    return 1;
}

void print_matrix(Matrix* matrix) {
    printf("Rows %d, Columns %d\n", matrix->rows, matrix->columns);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            double value = matrix->values[_get_index(matrix, i, j)];
            printf("%6.2f", value);
        }
        printf("\n");
    }
}

Matrix* transpose_matrix(Matrix* matrix) {
    Matrix* transposed = create_matrix(matrix->columns, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            double value = get_matrix_value(matrix, i, j);
            set_matrix_value(transposed, j, i, value);
        }
    }
    return transposed;
}

Matrix* add_matrices(Matrix* m1, Matrix* m2) {
    if (m1->columns != m2->columns || m1->rows != m2->rows) {
        fprintf(stderr, "Incorrect dimensions for matrix addition\n");
        exit(EXIT_FAILURE);
    }
    int rows = m1->rows;
    int cols = m1->columns;
    Matrix* sum = create_matrix(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double d1 = get_matrix_value(m1, i, j);
            double d2 = get_matrix_value(m2, i, j);
            set_matrix_value(sum, i, j, d1 + d2);
        }
    }
    return sum;
}

Matrix* subtract_matrices(Matrix* m1, Matrix* m2) {
    if (m1->columns != m2->columns || m1->rows != m2->rows) {
        fprintf(stderr, "Incorrect dimensions for matrix subtraction\n");
        exit(EXIT_FAILURE);
    }
    int rows = m1->rows;
    int cols = m1->columns;
    Matrix* sum = create_matrix(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double d1 = get_matrix_value(m1, i, j);
            double d2 = get_matrix_value(m2, i, j);
            set_matrix_value(sum, i, j, d1 - d2);
        }
    }
    return sum;
}

Matrix* hadamard_product(Matrix* m1, Matrix* m2) {
    if (m1->columns != m2->columns || m1->rows != m2->rows) {
        fprintf(stderr, "Incorrect dimensions for matrix subtraction\n");
        exit(EXIT_FAILURE);
    }
    int rows = m1->rows;
    int cols = m1->columns;
    Matrix* sum = create_matrix(rows, cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double d1 = get_matrix_value(m1, i, j);
            double d2 = get_matrix_value(m2, i, j);
            set_matrix_value(sum, i, j, d1 * d2);
        }
    }
    return sum;
}


Matrix* element_wise_operation(Matrix* matrix, double (*func)(double)) {
    Matrix* result = create_matrix(matrix->rows, matrix->columns);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            double v = get_matrix_value(matrix, i, j);
            double resultValue = func(v);
            _set_matrix_value(result, i, j, resultValue);
        }
    }
    return result;
}

Matrix* scalar_multiply_matrix(Matrix* matrix, double scalar) {
    Matrix* product = create_matrix(matrix->rows, matrix->columns);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->columns; j++) {
            double v = get_matrix_value(matrix, i, j);
            _set_matrix_value(product, i, j, v * scalar);
        }
    }
    return product;
}

double _get_matrix_product_at(Matrix* m1, Matrix* m2, int row, int column) {
    int sum = 0;
    for (int i = 0; i < m1->columns; i++) {
        sum += get_matrix_value(m1, row, i) * get_matrix_value(m2, i, column);
    }
    return sum;
}

Matrix* multiply_matrices(Matrix* m1, Matrix* m2) {
    if (m1->columns != m2->rows) {
        fprintf(stderr, "Incorrect dimensions for matrix multiplication\n");
        exit(EXIT_FAILURE);
    }
    Matrix* product = create_matrix(m1->rows, m2->columns);
    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m2->columns; j++) {
            double value = _get_matrix_product_at(m1, m2, i, j);
            set_matrix_value(product, i, j, value);
        }
    }
    return product;
}