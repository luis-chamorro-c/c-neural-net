#ifndef MATRICES_H
#define MATRICES_H

#include <stdint.h>
#include <stdalign.h>

#define ALIGN_UP(size, align) (((size) + alignof(align) - 1) & ~(alignof(align) - 1))

typedef struct { 
    double* values;
    int columns;
    int rows;
} Matrix;

Matrix* create_matrix(int rows, int columns);

Matrix* create_matrices(int rows, int columns, int num_matrices);

Matrix* create_matrix_with_values(int rows, int columns, double* values);

void free_matrix(Matrix* matrix);

void free_matrices(Matrix** matrices, int num_matrices);

int set_matrix_value(Matrix* matrix, int row, int column, double value);

double get_matrix_value(Matrix* matrix, int row, int column);

int set_matrix_values(Matrix* matrix, int row, double* values, int length);

void print_matrix(Matrix* matrix);

Matrix* transpose_matrix(Matrix* matrix);

void add_matrices(Matrix* m1, Matrix* m2, Matrix* out);

void subtract_matrices(Matrix* m1, Matrix* m2, Matrix* out);

void hadamard_product(Matrix* m1, Matrix* m2, Matrix* out);

void element_wise_operation(Matrix* matrix, double (*func)(double), Matrix* out);

Matrix* scalar_multiply_matrix(Matrix* matrix, double scalar);

Matrix* multiply_matrices(Matrix* m1, Matrix* m2);

double get_cost(Matrix* m1, Matrix* m2);

#endif