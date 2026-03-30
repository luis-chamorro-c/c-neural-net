#include "matrices.h"

typedef struct {
    Matrix** weights;
    Matrix** biases;
    int* layers;
    int num_layers;
} Network;

Network *initialize_network(int* layers, int num_layers);

void free_network(Network* network);

void feed_forward_for_backprop(Network* network, Matrix* input, Matrix** activations, Matrix** pre_activations);

void backpropagation(Network* network, Matrix* input, Matrix* output, Matrix*** delta_w, Matrix*** delta_b);