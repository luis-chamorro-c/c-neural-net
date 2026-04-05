#ifndef READ_FILE_H

#include <stdint.h>
#include "matrices.h"

Matrix** read_label_file(char* file_name, int* file_size);

uint8_t* read_labels_as_int(char* file_name, int* file_size);

Matrix** read_images(char* file_name);

uint8_t convert_label_to_number(Matrix* label_matrix);

#endif