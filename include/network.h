#ifndef NETWORK_H
#define NETWORK_H

#include "matrices.h"

typedef struct {
    Matrix* weights;
    Matrix* biases;
    int* layers;
    int num_layers;
} Network;

Network *initialize_network(int* layers, int num_layers);

void free_network(Network* network);

void feed_forward_for_backprop(Network* network, Matrix* input, Matrix** activations, Matrix** pre_activations);

void backpropagation(Network* network, Matrix* input, Matrix* output, Matrix*** delta_w, Matrix*** delta_b);

void update_with_samples(Network *network, Matrix *input, Matrix *output, double learning_rate, int start_index);

Matrix* feed_forward(Network* network, Matrix* input);

#endif