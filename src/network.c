#include "matrices.h"
#include "network.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

double rand_normal() {
    // Generate two uniform random numbers (0,1]
    double u1 = (rand() + 1.0) / (RAND_MAX + 1.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 1.0);

    // Box-Muller transform
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

Network *initialize_network(int* layers, int num_layers) {
    Network *network = malloc(sizeof(Network));
    network->num_layers = num_layers;

    int *layers_copy = malloc(sizeof(int) * num_layers);
    memcpy(layers_copy, layers, sizeof(int) * num_layers);
    network->layers = layers_copy;

    Matrix **curr_weights = malloc(sizeof(Matrix*) * num_layers);
    for (int i = 0; i < num_layers - 1; i++) {
        int l1_count = layers[i];
        int l2_count = layers[i + 1];
        Matrix* curr_matrix = create_matrix(l2_count, l1_count);
        for (int j = 0; j < l2_count; j++) {
            for (int k = 0; k < l1_count; k++) {
                set_matrix_value(curr_matrix, j, k, rand_normal());
            }
        }
        curr_weights[i] = curr_matrix;
    }
    network->weights = curr_weights;

    Matrix **biases = malloc(sizeof(Matrix*) * num_layers);
    for (int i = 0; i < num_layers - 1; i++) {
        int l_count = layers[i+1];
        Matrix *curr_bias = create_matrix(l_count, 1);
        for (int i = 0; i < l_count; i++) {
            set_matrix_value(curr_bias, i, 0, rand_normal());
        }
        biases[i] = curr_bias;
    }
    network->biases = biases;
    return network;
}

void free_network(Network* network) {
    for (int i = 0; i < network->num_layers-1; i++) {
        Matrix* weight = network->weights[i];
        free_matrix(weight);
        Matrix* bias = network->biases[i];
        free_matrix(bias);
    }
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

Matrix* cost_derivative(Matrix* activations, Matrix* y) {
    return subtract_matrices(activations, y);
}

void backpropagation(Network* network, Matrix* input, Matrix* output, Matrix*** output_delta_w, Matrix*** output_delta_b) {
    int size = network->num_layers - 1;
    Matrix** activations = malloc(sizeof(Matrix*) * network->num_layers);
    activations[0] = input;
    Matrix** pre_activations = malloc(sizeof(Matrix*) * size);
    
    feed_forward_for_backprop(network, input, activations, pre_activations);

    // Get error for last layer
    Matrix* cost_der = cost_derivative(activations[size - 1], output);
    Matrix* sigmoid_der = element_wise_operation(pre_activations[size - 1], sigmoid_prime);
    Matrix* error = hadamard_product(cost_der, sigmoid_der);
    free_matrix(cost_der);
    free_matrix(sigmoid_der);

    // Compute partial derivative of bias and weights for last layer
    Matrix** delta_b = malloc(sizeof(Matrix*) * size);
    Matrix** delta_w = malloc(sizeof(Matrix*) * size);

    delta_b[size - 1] = error;
    Matrix* activation_t = transpose_matrix(activations[size - 1]);
    delta_w[size - 1] = multiply_matrices(error, activation_t);
    free_matrix(activation_t);

    // Compute partial derivative of bias and weights for every other layer
    for (int i = size - 2; i >= 0; i--) {
        Matrix* pre_activation = pre_activations[i];
        Matrix* sig_dev_pre_activation = element_wise_operation(pre_activation, sigmoid_prime);
        Matrix* weight_t = transpose_matrix(network->weights[i + 1]);
        Matrix* weight_x_error = multiply_matrices(weight_t, error);
        error = hadamard_product(weight_x_error, sig_dev_pre_activation);
        free_matrix(sig_dev_pre_activation);
        free_matrix(weight_t);
        free_matrix(weight_x_error);

        delta_b[i] = error;
        Matrix* activ_t = transpose_matrix(activations[i - 1]);
        delta_w[i] = multiply_matrices(error, activ_t);
        free_matrix(activ_t);
    }

    // Clean up
    free_matrices(pre_activations, size);
    free_matrices(activations, network->num_layers);

    // Return
    (*output_delta_b) = delta_b;
    (*output_delta_w) = delta_w;
}

void feed_forward_for_backprop(Network* network, Matrix* input, Matrix** activations, Matrix** pre_activations) {
    Matrix* current = input;
    for (int i = 0; i < network->num_layers - 1; i++) {
        Matrix* weights = network->weights[i];
        Matrix* biases = network->biases[i];
        
        Matrix* product = multiply_matrices(weights, current);
        Matrix* sum = add_matrices(product, biases);
        pre_activations[i] = sum;
        free_matrix(product);
    
        Matrix* sigmoid_m = element_wise_operation(sum, sigmoid);
        activations[i] = sigmoid_m;
        current = sigmoid_m;
    }
}