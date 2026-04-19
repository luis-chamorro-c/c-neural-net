#include "matrix_arena.h"
#include "matrices.h"
#include <stdalign.h>
#include <string.h>

MatArena *allocate_arena(size_t capacity) {
    MatArena *arena = malloc(sizeof(MatArena));
    arena->capacity = capacity;
    arena->start_pos = malloc(capacity);
    arena->curr_pos = 0;
    return arena;
}

void free_arena(MatArena *arena) {
    free(arena->start_pos);
    free(arena);
}

Matrix* allocate_matrix(MatArena *arena, int rows, int columns) {
    size_t alloc_start = ALIGN_UP(arena->curr_pos, Matrix);
    Matrix *m = (Matrix*)(arena->start_pos + alloc_start);
    m->rows = rows;
    m->columns = columns;
    arena->curr_pos = alloc_start + sizeof(Matrix);
    
    size_t float_start = ALIGN_UP(arena->curr_pos, float);
    m->values = (float*)(arena->start_pos + float_start);
    arena->curr_pos = float_start + (sizeof(float) * rows * columns);
    return m;
}

Matrix* allocate_matrices(MatArena *arena, int* rows, int* columns, int num_matrices) {
    size_t alloc_start = ALIGN_UP(arena->curr_pos, Matrix);
    Matrix* matrix_arr = (Matrix*)(arena->start_pos + alloc_start);

    arena->curr_pos = alloc_start + (sizeof(Matrix) * num_matrices);
    arena->curr_pos = ALIGN_UP(arena->curr_pos, float);
    
    size_t float_start = arena->curr_pos;
    uint8_t *curr_ptr = arena->start_pos + arena->curr_pos;
    for (int i = 0; i < num_matrices; i++) {
        matrix_arr[i].rows = rows[i];
        matrix_arr[i].columns = columns[i];
        matrix_arr[i].values = (float*)(curr_ptr);

        size_t to_append = sizeof(float) * rows[i] * columns[i];
        memset(curr_ptr, 0, to_append);
        arena->curr_pos += to_append;
        curr_ptr += to_append;
    }
    return matrix_arr;
}

void clear_arena(MatArena *arena) {
    arena->curr_pos = 0;
}