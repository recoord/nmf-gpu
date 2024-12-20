#include <cublas.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <string>


class Matrix {
    public:
    float *data = nullptr;
    uint32_t rows;
    uint32_t cols;

    void clone_to_padded(Matrix* padded);

    Matrix(uint32_t rows, uint32_t cols);
    Matrix(float *host_data, uint32_t rows, uint32_t cols);
    Matrix(float value, uint32_t rows, uint32_t cols);
    ~Matrix();
};

typedef enum { compute, cleanup } action_t;

// creating, allocating, moving matrices
void copy_matrix_on_device(Matrix A, Matrix B);
void copy_from_padded(Matrix A, Matrix Apad);

// Matrix analysis
float nan_check_d(action_t action, Matrix a, uint32_t *params);
float zero_check_d(action_t action, Matrix a, uint32_t *params);
float zero_check(Matrix a);

// sgemms
void matrix_multiply_d(Matrix a, Matrix b, Matrix c);
void matrix_multiply_AtB_d(Matrix a, Matrix b, Matrix c);
void matrix_multiply_ABt_d(Matrix a, Matrix b, Matrix c);

// element operations
void element_multiply_d(Matrix a, Matrix b, Matrix c, uint32_t block_size);
void element_divide_d(Matrix a, Matrix b, Matrix c, uint32_t block_size);
void matrix_eps_d(Matrix a, uint32_t block_size, cudaStream_t stream);

// row/col-wise
void row_divide_d(Matrix a, Matrix b, Matrix c);
void col_divide_d(Matrix a, Matrix b, Matrix c);
void sum_cols_d(action_t action, Matrix a, Matrix c, uint32_t *params);
void sum_rows_d(action_t action, Matrix a, Matrix c, uint32_t *params);
