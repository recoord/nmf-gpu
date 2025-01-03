#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <string>

#define PAD_MULT 32

class Memory {
    public:
    float *data = nullptr;
    uint32_t size = 0;

    Memory(uint32_t size);
    ~Memory();
};

class Matrix {
    private:
    void add_padding();
    void ensure_memory(uint32_t size);

    public:
    float *data = nullptr;

    uint32_t rows;
    uint32_t cols;
    uint32_t rows_padded;
    uint32_t cols_padded;

    void set_epsilon(uint32_t block_size, cudaStream_t stream);
    void sum_cols(Matrix *output, Memory *aux, uint32_t *params, cudaStream_t stream);
    void sum_rows(Matrix *output, Memory *aux, uint32_t *params, cudaStream_t stream);

    Matrix(uint32_t rows, uint32_t cols);
    Matrix(float *host_data, uint32_t rows, uint32_t cols, cudaStream_t stream);
    Matrix(float value, uint32_t rows, uint32_t cols, cudaStream_t stream);
    ~Matrix();
};

// sgemms
void matrix_multiply(Matrix *a, Matrix *b, Matrix *c, cublasHandle_t handle);
void matrix_multiply_AtB(Matrix *a, Matrix *b, Matrix *c, cublasHandle_t handle);
void matrix_multiply_ABt(Matrix *a, Matrix *b, Matrix *c, cublasHandle_t handle);

// element operations
void element_multiply(Matrix *a, Matrix *b, Matrix *c, uint32_t block_size, cudaStream_t stream);
void element_divide(Matrix *a, Matrix *b, Matrix *c, uint32_t block_size, cudaStream_t stream);

// row/col-wise
void row_divide(Matrix *a, Matrix *b, Matrix *c, cudaStream_t stream);
void col_divide(Matrix *a, Matrix *b, Matrix *c, cudaStream_t stream);
