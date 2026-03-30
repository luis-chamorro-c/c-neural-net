#ifndef READ_FILE_H

#include <stdint.h>
#include "matrices.h"

Matrix** read_label_file(char* file_name);

Matrix** read_images(char* file_name);

#endif