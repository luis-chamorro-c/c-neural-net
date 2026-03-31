#ifndef MATRICES_H

#include <stdint.h>

typedef struct Matrix Matrix;

Matrix* create_matrix(int rows, int columns);

Matrix* create_matrix_with_values(int rows, int columns, double* values);

void free_matrix(Matrix* matrix);

void free_matrices(Matrix** matrices, int num_matrices);

int set_matrix_value(Matrix* matrix, int row, int column, double value);

double get_matrix_value(Matrix* matrix, int row, int column);

int set_matrix_values(Matrix* matrix, int row, double* values, int length);

void print_matrix(Matrix* matrix);

Matrix* transpose_matrix(Matrix* matrix);

Matrix* add_matrices(Matrix* m1, Matrix* m2);

Matrix* subtract_matrices(Matrix* m1, Matrix* m2);

Matrix* hadamard_product(Matrix* m1, Matrix* m2);

Matrix* element_wise_operation(Matrix* matrix, double (*func)(double));

Matrix* scalar_multiply_matrix(Matrix* matrix, double scalar);

Matrix* multiply_matrices(Matrix* m1, Matrix* m2);

double get_cost(Matrix* m1, Matrix* m2);

#endif