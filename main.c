#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrices.h"
#include "read_file.h"
#include "network.h"

void matrix_test() {
  double myVals[12] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5};
  Matrix* m1 = create_matrix_with_values(2, 3, myVals);
  
  print_matrix(m1);
  free_matrix(m1);
}

void read_file(Matrix*** input_img, Matrix*** input_label) {
  char* filename = "./mnist-dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte";
  Matrix** labels = read_label_file(filename);
  
  char* image_filename = "./mnist-dataset/train-images-idx3-ubyte/train-images-idx3-ubyte";
  Matrix** images = read_images(image_filename);

  (*input_img) = images;
  (*input_label) = labels;
}

int main() {
  srand(0);

  int layers[3] = { (28*28), 32, 10 };
  Network *network = initialize_network(layers, 3);

  Matrix** input;
  Matrix** output;
  read_file(&input, &output);
  
  // backpropagation(network, input, )

  // free_matrices(input, 60000);
  // free_matrices(output, 60000);
  // free_network(network);
}