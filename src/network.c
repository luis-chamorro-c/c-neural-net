#include "matrices.h"
#include "network.h"
#include "matrix_arena.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdalign.h>

const static int SAMPLE_SIZE = 10;

double fake_random(int j, int k) {
    // I debugged this by comparing to the python version, so I needed to initialize with a deterministic but
    // random looking set of numbers that I could compare 1 to 1. This was a pretty good estimate
    double val = sin(j * 12.9898 + k * 78.233) * 43758.5453;
    return (val - floor(val)) - 0.5;
}

Network *initialize_network(int* layers, int num_layers) {
    Network *network = malloc(sizeof(Network));
    network->num_layers = num_layers;

    int *layers_copy = malloc(sizeof(int) * num_layers);
    memcpy(layers_copy, layers, sizeof(int) * num_layers);
    network->layers = layers_copy;

    int struct_size = sizeof(Matrix) * (num_layers - 1);
    size_t aligned_struct_size = ALIGN_UP(struct_size, double);

    int num_entries = 0;
    int layers_sum = 0;
    for (int i = 0; i < num_layers - 1; i++) {
        num_entries += layers[i] * layers[i+1];
        layers_sum += layers[i+1];
    }
    
    Matrix* curr_weights = malloc(aligned_struct_size + (sizeof(double) * num_entries));
    double* values_start = (double*)((uint8_t*)curr_weights + aligned_struct_size);
    double* curr_values = values_start;
    for (int i = 0; i < num_layers - 1; i++) {
        int l1_count = layers[i];
        int l2_count = layers[i + 1];
        
        curr_weights[i].rows = l2_count;
        curr_weights[i].columns = l1_count;
        curr_weights[i].values = curr_values;

        int offset = l1_count * l2_count;
        curr_values += offset;

        for (int j = 0; j < offset; j++) {
            curr_weights[i].values[j] = fake_random(j / l1_count, j % l1_count);
        }
    }
    network->weights = curr_weights;

    Matrix *biases = malloc(aligned_struct_size + sizeof(double) * layers_sum);
    double* biases_start = (double*)((uint8_t*)biases + aligned_struct_size);
    double* last_biases = biases_start;
    for (int i = 0; i < num_layers - 1; i++) {
        biases[i].rows = layers[i+1];
        biases[i].columns = 1;
        biases[i].values = last_biases;
        memset(biases[i].values, 0, layers[i+1]);

        last_biases += layers[i+1];
    }
    network->biases = biases;
    return network;
}

void free_network(Network* network) {
    free(network->weights);
    free(network->biases);
    free(network->layers);
    free(network);
}

double sigmoid(double input) {
    return 1/(1+exp(-input));
}

double sigmoid_prime(double input) {
    return sigmoid(input) * (1-sigmoid(input));
}

void feed_forward_for_backprop(Network* network, Matrix* input, Matrix* activations, Matrix* pre_activations) {
    Matrix* current = input;
    for (int i = 0; i < network->num_layers - 1; i++) {
        Matrix *weights = &network->weights[i];
        Matrix *biases = &network->biases[i];
        
        Matrix* intermediate = multiply_matrices(weights, current);
        add_matrices(intermediate, biases, &pre_activations[i]);
    
        element_wise_operation(intermediate, sigmoid, &activations[i+1]);
        current = &activations[i+1];
    }
}

Matrix* feed_forward(Network* network, Matrix* input) {
    Matrix* current = input;
    for (int i = 0; i < network->num_layers - 1; i++) {
        Matrix *weights = &network->weights[i];
        Matrix *biases = &network->biases[i];
        
        Matrix* intermediate = multiply_matrices(weights, current);
        add_matrices(intermediate, biases, intermediate);
    
        element_wise_operation(intermediate, sigmoid, intermediate);
        Matrix* prev_current = current;
        current = intermediate;
        if (i > 0) {
            free_matrix(prev_current);
        }
    }
    return current;
}

void backpropagation(MatArena *arena, Network* network, Matrix* input, Matrix* output, Matrix*** output_delta_w, Matrix*** output_delta_b) {
    int size = network->num_layers - 1;
    int columns[network->num_layers];
    for (int i = 0; i < network->num_layers; i++) {
        columns[i] = 1;
    }
    Matrix *activations = allocate_matrices(arena, network->layers, columns, network->num_layers);
    activations[0] = *input;
    Matrix* pre_activations = allocate_matrices(arena, network->layers + 1, columns, size);
    
    feed_forward_for_backprop(network, input, activations, pre_activations);

    // Get error for last layer
    Matrix *error = allocate_matrix(arena, output->rows, output->columns);
    Matrix *sigmoid_der = allocate_matrix(arena, output->rows, output->columns);

    subtract_matrices(&activations[network->num_layers - 1], output, error);
    element_wise_operation(&pre_activations[size - 1], sigmoid_prime, sigmoid_der);
    hadamard_product(error, sigmoid_der, error);

    // Compute partial derivative of bias and weights for last layer
    Matrix** delta_b = malloc(sizeof(Matrix*) * size);
    Matrix** delta_w = malloc(sizeof(Matrix*) * size);

    delta_b[size - 1] = error;
    Matrix* activation_t = transpose_matrix(&activations[size - 1]);
    delta_w[size - 1] = multiply_matrices(error, activation_t);
    free_matrix(activation_t);

    // Compute partial derivative of bias and weights for every other layer
    for (int i = size - 2; i >= 0; i--) {
        Matrix* pre_activation = &pre_activations[i];
        Matrix* sig_dev_pre_activation = allocate_matrix(arena, pre_activation->rows, pre_activation->columns);
        element_wise_operation(pre_activation, sigmoid_prime, sig_dev_pre_activation);
        Matrix* weight_t = transpose_matrix(&network->weights[i + 1]);
        Matrix* weight_x_error = multiply_matrices(weight_t, error);
        Matrix* new_error = allocate_matrix(arena, sig_dev_pre_activation->rows, sig_dev_pre_activation->columns);
        hadamard_product(weight_x_error, sig_dev_pre_activation, new_error);
        error = new_error;
        
        free_matrix(weight_t);
        free_matrix(weight_x_error);

        delta_b[i] = error;
        Matrix* activ_t = transpose_matrix(&activations[i]);
        delta_w[i] = multiply_matrices(error, activ_t);
        free_matrix(activ_t);
    }

    // Return
    (*output_delta_b) = delta_b;
    (*output_delta_w) = delta_w;
}

void update_with_samples(MatArena *arena, Network *network, Matrix *input, Matrix *output, double learning_rate, int start_index) {
    Matrix *delta_w_sums = allocate_matrices(arena, (network->layers + 1), network->layers, network->num_layers - 1);
    int columns[network->num_layers -1];
    for (int i = 0; i < network->num_layers; i++) {
        columns[i] = 1;
    }
    Matrix *delta_b_sums = allocate_matrices(arena, (network->layers + 1), columns, network->num_layers - 1);

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        Matrix** delta_w;
        Matrix** delta_b;
        backpropagation(arena, network, &input[start_index + i], &output[start_index + i], &delta_w, &delta_b);
        
        for (int i = 0; i < network->num_layers - 1; i++) {
            Matrix* curr_w_sum = &delta_w_sums[i];
            add_matrices(curr_w_sum, delta_w[i], curr_w_sum);

            Matrix* curr_b_sum = &delta_b_sums[i];
            add_matrices(curr_b_sum, delta_b[i], curr_b_sum);
        }

        free_matrices(delta_w, network->num_layers-1);
    }

    double to_mult = (learning_rate/SAMPLE_SIZE);
    for (int i = 0; i < network->num_layers - 1; i++) {
        Matrix* w_multiplied = scalar_multiply_matrix(&delta_w_sums[i], to_mult);
        subtract_matrices(&network->weights[i], w_multiplied, &network->weights[i]);

        free_matrix(w_multiplied);

        Matrix* b_multiplied = scalar_multiply_matrix(&delta_b_sums[i], to_mult);
        subtract_matrices(&network->biases[i], b_multiplied, &network->biases[i]);
        free_matrix(b_multiplied);
    }
}
