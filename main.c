#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrices.h"
#include "read_file.h"
#include "network.h"
#include <time.h>
#include "matrix_arena.h"

#define BASE_DIR "./mnist-dataset"

#define TRAINING_LABELS BASE_DIR"/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
#define TRAINING_IMAGES BASE_DIR"/train-images-idx3-ubyte/train-images-idx3-ubyte"

#define TESTING_LABELS BASE_DIR"/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
#define TESTING_IMAGES BASE_DIR"/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"

#define NETWORK_FILE "./stored_networks/network"

void matrix_test() {
  MatArena *arena = allocate_arena(MB(5));

  int rows[] = { 3, 5, 1};
  int columns[] = { 2, 5, 4 };

  Matrix* m_arr = allocate_matrices(arena, rows, columns, 3);
  
  double count = 0;
  for (int i = 0; i < 3; i++) {
    Matrix *m = &m_arr[i];
    for (int j = 0; j < m->rows; j++) {
      for (int k = 0; k < m->columns; k++) {
        set_matrix_value(m, j, k, count);
        count++;
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    print_matrix(&m_arr[i]);
  }

  free_arena(arena);
}

void read_training_files(Matrix** input_img, Matrix** input_label, int *input_size) {
  int size;
  Matrix* labels = read_label_file(TRAINING_LABELS, &size);
  Matrix* images = read_images(TRAINING_IMAGES);

  (*input_img) = images;
  (*input_label) = labels;
  (*input_size) = size;
}

void read_test_files(Matrix** input_img, uint8_t **input_label, int* input_size) {
  int size;
  uint8_t* labels = read_labels_as_int(TESTING_LABELS, &size);
  Matrix* images = read_images(TESTING_IMAGES);

  (*input_img) = images;
  (*input_label) = labels;
  (*input_size) = size;
}

void shuffle_training_data(Matrix* inputs, Matrix* outputs, int training_count) {
  for (int i = training_count - 1; i >= 0; i--) {
    int rand_index = rand() % (i + 1);
    Matrix input_temp = inputs[i];
    inputs[i] = inputs[rand_index];
    inputs[rand_index] = input_temp;

    Matrix output_temp = outputs[i];
    outputs[i] = outputs[rand_index];
    outputs[rand_index] = output_temp;
  }
}

Network* train(int num_epochs) {
  printf("Training has begun\n");
  clock_t start = clock();

  int layers[3] = { (28*28), 30, 10 };
  Network *network = initialize_network(layers, 3);

  Matrix* input;
  Matrix* output;
  int training_count;
  read_training_files(&input, &output, &training_count);
  
  MatArena *arena = allocate_arena(MB(30));
  for (int i = 0; i < num_epochs; i++) {
    for (int j = 0; j < training_count; j+=10) {
      if (j % 10000 == 0) {
        printf("Trained %d entries\n", i * training_count + j);
      }
      update_with_samples(arena, network, input, output, 0.5, j);
      clear_arena(arena);
    }
    shuffle_training_data(input, output, training_count);
  }
  free_arena(arena);

  free(input);
  free(output);

  clock_t end = clock();
  printf("Training complete! Training %d samples took %.2f seconds\n", training_count, (double)(end - start) / CLOCKS_PER_SEC);
  return network;
}

void measure_performance(Network* network) {
  printf("Testing started\n");
  Matrix* input;
  uint8_t *output;
  int total_samples;
  read_test_files(&input, &output, &total_samples);

  int success = 0;
  MatArena *arena = allocate_arena(MB(5));
  for (int i = 0; i < total_samples; i++) {
    Matrix* result = feed_forward(arena, network, &input[i]);
    uint8_t num_result = convert_label_to_number(result);
    if (num_result == output[i]) {
      success++;
    }
    clear_arena(arena);
  }

  free(input);
  free(output);

  double success_rate = ((double)success / total_samples) * 100;
  printf("Testing completed! Tested %d samples. Success rate: %.2f%%\n", total_samples, success_rate);
}

int main(int argc, char *argv[]) {
  int num_epochs = 1;
  if (argc > 1) {
    num_epochs = atoi(argv[1]);
  }
  Network *network = train(num_epochs);
  measure_performance(network);
  save_network_to_file(network, NETWORK_FILE);
  free_network(network);
}