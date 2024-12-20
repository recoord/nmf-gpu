
#include <cublas.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <string>


class Matrix {
    public:
    float *data;
    uint32_t rows;
    uint32_t cols;

    void clone_to_padded(Matrix* padded, cudaStream_t stream);

    Matrix(uint32_t rows, uint32_t cols, cudaStream_t stream);
    Matrix(float value, uint32_t rows, uint32_t cols, cudaStream_t stream);
    Matrix(float *host_data, uint32_t rows, uint32_t cols, cudaStream_t stream);
    ~Matrix();
};

// typedef struct {
//     float *data;
//     uint32_t rows;
//     uint32_t cols;
// } matrix;

typedef enum { compute, cleanup } action_t;

// creating, allocating, moving matrices
Matrix read_matrix(std::string file, cudaStream_t stream);
void write_matrix(Matrix A, std::string file);
// void create_matrix_on_device(Matrix *A, int rows, int cols, float value);
void copy_matrix_on_device(Matrix A, Matrix B);
// void copy_to_padded(Matrix A, Matrix Apad);
// void copy_matrix_to_device_padded(Matrix A, Matrix Apad);
void copy_from_padded(Matrix A, Matrix Apad);
// void allocate_matrix_on_device(Matrix *A);
// void free_matrix_on_device(Matrix *A);
// void destroy_matrix(Matrix *A);

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
