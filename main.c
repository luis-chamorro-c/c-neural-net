#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrices.h"
#include "read_file.h"
#include "network.h"
#include <time.h>

#define BASE_DIR "/Users/luischamorro/repo/c-neural-net/mnist-dataset"

#define TRAINING_LABELS BASE_DIR"/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
#define TRAINING_IMAGES BASE_DIR"/train-images-idx3-ubyte/train-images-idx3-ubyte"

#define TESTING_LABELS BASE_DIR"/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
#define TESTING_IMAGES BASE_DIR"/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"

void matrix_test() {
  double myVals[12] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5};
  Matrix* m1 = create_matrix_with_values(2, 3, myVals);
  
  print_matrix(m1);
  free_matrix(m1);
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

Network* train() {
  printf("Training has begun\n");
  clock_t start = clock();

  int layers[3] = { (28*28), 30, 10 };
  Network *network = initialize_network(layers, 3);

  Matrix* input;
  Matrix* output;
  int training_count;
  read_training_files(&input, &output, &training_count);
  
  for (int i = 0; i < training_count; i+=10) {
    if (i % 10000 == 0) {
      printf("Trained %d entries\n", i);
    }
    update_with_samples(network, input, output, 0.5, i);
  }

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
  for (int i = 0; i < total_samples; i++) {
    Matrix* result = feed_forward(network, &input[i]);
    uint8_t num_result = convert_label_to_number(result);
    if (num_result == output[i]) {
      success++;
    }
    free_matrix(result);
  }

  free(input);
  free(output);

  double success_rate = ((double)success / total_samples) * 100;
  printf("Testing completed! Tested %d samples. Success rate: %.2f%%\n", total_samples, success_rate);
}

int main() {
  Network *network = train();
  measure_performance(network);
  free_network(network);
}