#ifndef MATRIX_ARENA_H
#define MATRIX_ARENA_H

#include "matrices.h"
#include <stdlib.h>

#define MB(x) ((size_t)(x) << 20)

typedef struct {
    uint8_t *start_pos;
    size_t curr_pos;
    size_t capacity;
} MatArena;

MatArena *allocate_arena(size_t capacity);

void free_arena(MatArena *arena);

Matrix* allocate_matrix(MatArena *arena, int rows, int columns);

Matrix* allocate_matrices(MatArena *arena, int* rows, int* columns, int num_matrices);

void clear_arena(MatArena *arena);

#endif