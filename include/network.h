#ifndef NETWORK_H
#define NETWORK_H

#include "matrices.h"
#include "matrix_arena.h"

typedef struct {
    Matrix* weights;
    Matrix* biases;
    int* layers;
    int num_layers;
} Network;

Network *initialize_network(int* layers, int num_layers);

void free_network(Network* network);

void update_with_samples(MatArena *arena, Network *network, Matrix *input, Matrix *output, double learning_rate, int start_index);

Matrix* feed_forward(MatArena *arena, Network* network, Matrix* input);

void save_network_to_file(Network *network, char *file_name);

#endif