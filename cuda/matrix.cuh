
#include <cublas.h>
#include <math.h>
#include <string>


typedef struct {
    float *mat;   // pointer to host data
    float *mat_d; // pointer to device data
    int dim[2];   // dimensions: {rows,cols}
} matrix;

typedef enum { compute, cleanup } action_t;

// creating, allocating, moving matrices
matrix read_matrix(std::string file, cudaStream_t stream);
void write_matrix(matrix A, std::string file);
void create_matrix(matrix *A, int rows, int cols, float value);
void create_matrix_on_device(matrix *A, int rows, int cols, float value);
void create_matrix_on_both(matrix *A, int rows, int cols, float value);
void copy_matrix_to_device(matrix *A, cudaStream_t stream);
void copy_matrix_on_device(matrix A, matrix B);
void copy_to_padded(matrix A, matrix Apad);
void copy_matrix_to_device_padded(matrix A, matrix Apad);
void copy_from_padded(matrix A, matrix Apad);
void allocate_matrix_on_device(matrix *A);
void free_matrix_on_device(matrix *A);
void destroy_matrix(matrix *A);

// matrix analysis
float nan_check_d(action_t action, matrix a, int *params);
float zero_check_d(action_t action, matrix a, int *params);
float zero_check(matrix a);

// sgemms
void matrix_multiply_d(matrix a, matrix b, matrix c);
void matrix_multiply_AtB_d(matrix a, matrix b, matrix c);
void matrix_multiply_ABt_d(matrix a, matrix b, matrix c);

// element operations
void element_multiply_d(matrix a, matrix b, matrix c, int block_size);
void element_divide_d(matrix a, matrix b, matrix c, int block_size);
void matrix_eps_d(matrix a, int block_size, cudaStream_t stream);

// row/col-wise
void row_divide_d(matrix a, matrix b, matrix c);
void col_divide_d(matrix a, matrix b, matrix c);
void sum_cols_d(action_t action, matrix a, matrix c, int *params);
void sum_rows_d(action_t action, matrix a, matrix c, int *params);
