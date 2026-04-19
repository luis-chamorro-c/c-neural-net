#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "matrices.h"
#include "read_file.h"

static const int LABEL_FILE_TYPE = 2049;
static const int IMAGE_FILE_TYPE = 2051;

int swap_int(int val) {
    uint32_t u = (uint32_t)val;
    u = (u >> 24) |
        ((u >> 8) & 0x0000FF00) |
        ((u << 8) & 0x00FF0000) |
        (u << 24);
    return (int)u;
}

uint8_t convert_label_to_number(Matrix* label_matrix) {
    float max = -1;
    uint8_t max_num = 0;
    for (int i = 0; i < 10; i++) {
        float matrix_value = get_matrix_value(label_matrix, i, 0);
        if (matrix_value > max) {
            max = matrix_value;
            max_num = i;
        }
    }
    return max_num;
}

uint8_t* read_labels_as_int(char* file_name, int* file_size) {
    FILE *fptr = fopen(file_name, "rb");
    if (fptr == NULL) {
        perror("Error opening label file");
        exit(EXIT_FAILURE);
    }
    int metadata[2];
    fread(&metadata, sizeof(int), 2, fptr);
    int file_type = swap_int(metadata[0]);
    if (file_type != LABEL_FILE_TYPE) {
        fprintf(stderr, "Did not receive correct label file type\n");
        exit(EXIT_FAILURE);
    }
    int arr_size = swap_int(metadata[1]);
    
    uint8_t *all_labels = malloc(sizeof(uint8_t) * arr_size);
    fread(all_labels, sizeof(uint8_t), arr_size, fptr);
    fclose(fptr);
    
    (*file_size) = arr_size;
    return all_labels;
}

Matrix* read_label_file(char* file_name, int* input_size) {
    int file_size;
    uint8_t *all_labels = read_labels_as_int(file_name, &file_size);

    Matrix* label_matrices = create_matrices(10, 1, file_size);
    for (int i = 0; i < file_size; i++) {
        label_matrices[i].rows = 10;
        label_matrices[i].columns = 1;
        int index = all_labels[i];
        label_matrices[i].values[index] = 1;
    }
    free(all_labels);
    (*input_size) = file_size;
    return label_matrices;
}

Matrix* read_images(char* file_name) {
    FILE *fptr = fopen(file_name, "rb");
    int metadata[4];
    fread(&metadata, sizeof(int), 4, fptr);
    
    int file_type = swap_int(metadata[0]);
    int num_images = swap_int(metadata[1]);
    int num_rows = swap_int(metadata[2]);
    int num_cols = swap_int(metadata[3]);

    if (file_type != IMAGE_FILE_TYPE) {
        fprintf(stderr, "Did not receive correct image file type\n");
    }

    Matrix *images = create_matrices(num_rows, num_cols, num_images);
    for (int i = 0; i < num_images; i++) {
        Matrix image = images[i];
        uint8_t values[num_rows * num_cols];
        fread(&values, sizeof(uint8_t), num_rows * num_cols, fptr);
        float *matrix_values = image.values;
        for (int j = 0; j < num_rows * num_cols; j++) {
            matrix_values[j] = (float)values[j] / 255;
        }
    }
    return images;
}