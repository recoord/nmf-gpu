#include <cassert>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "error-check.hpp"
#include "matrix.cuh"

#define ITER_CHECK 25     // status printed and convergence check every ITER_CHECK iterations
#define MAX_ITER 200      // max number of iterations
#define CONVERGE_THRESH 0 // set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise

void run_async(
    Matrix *W, Matrix *H, Matrix *X, const float thresh, const uint32_t max_iter, cublasHandle_t cublas_handle,
    cudaStream_t stream
);
void update_h(
    Matrix *W, Matrix *H, Matrix *X, Matrix *Z, Matrix *sumW, Matrix *WtZ, uint32_t *M_params, Memory *aux_memory,
    cublasHandle_t cublas_handle, cudaStream_t stream
);
void update_w(
    Matrix *W, Matrix *H, Matrix *X, Matrix *Z, Matrix *sumH2, Matrix *ZHt, uint32_t *N_params, Memory *aux_memory,
    cublasHandle_t cublas_handle, cudaStream_t stream
);
uint32_t nextpow2(uint32_t x);
Matrix read_matrix(std::string file, cudaStream_t stream);
void write_matrix(Matrix *matrix, std::string file, cudaStream_t stream);


int32_t main(int32_t argc, char *argv[]) {
    cudaStream_t stream;
    cudaAssert(cudaStreamCreate(&stream));
    cublasHandle_t cublas_handle;
    cudaAssert(cublasCreate(&cublas_handle));
    cudaAssert(cublasSetStream(cublas_handle, stream));

    Matrix X = read_matrix("../X.bin", stream);
    Matrix H = read_matrix("../H.bin", stream);
    Matrix W = read_matrix("../W.bin", stream);

    // Run iterative nmf minimization
    run_async(&W, &H, &X, CONVERGE_THRESH, MAX_ITER, cublas_handle, stream);

    write_matrix(&W, "../Wout.bin", stream);
    write_matrix(&H, "../Hout.bin", stream);

    cudaAssert(cublasDestroy(cublas_handle));
    cudaAssert(cudaStreamDestroy(stream));

    return 0;
}

void init_params(uint32_t value, uint32_t *params) {
    uint32_t padded_value = value;
    if(value % PAD_MULT != 0) {
        padded_value = value + (PAD_MULT - (value % PAD_MULT));
    }

    uint32_t rem;
    rem = nextpow2(padded_value / 128 + (!(padded_value % 128) ? 0 : 1));
    if(rem <= 128) {
        params[0] = 128;
        params[1] = rem;
    } else if(rem <= 512) {
        params[0] = rem;
        params[1] = 128;
    } else {
        fprintf(stderr, "reduction parameter error\n");
        exit(1);
    }

    params[2] = 1;
    params[3] = 1;
}

void run_async(
    Matrix *W, Matrix *H, Matrix *X, const float thresh, const uint32_t max_iter, cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    const uint32_t M = W->rows;
    const uint32_t K = W->cols;
    const uint32_t N = H->cols;

    // find reduction parameters
    uint32_t N_params[4]; // N size reductions (rows)
    uint32_t M_params[4]; // M size reductions (cols)

    init_params(N, N_params);
    init_params(M, M_params);

    Memory aux_memory(512); // auxiliary memory for summing rows/cols. The size should be dynamically allocated

    // initialize temporary matrices
    Matrix Z(0.0f, M, N, stream);     // Matrix to hold X./(W*H+EPS)
    Matrix WtZ(0.0f, K, N, stream);   // Matrix to hold W'*Z
    Matrix ZHt(0.0f, M, K, stream);   // Matrix to hold Z*H'
    Matrix sumW(0.0f, 1, K, stream);  // Matrix to hold sum(W) [sum of cols of W]
    Matrix sumH2(0.0f, K, 1, stream); // Matrix to hold sum(H,2) [sum of rows of H]

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    cudaAssert(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    // MatLab algorithm:
    // Z = X./(W*H+eps); H = H.*(W'*Z)./(repmat(sum(W)',1,F));
    // Z = X./(W*H+eps);
    // W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
    update_h(W, H, X, &Z, &sumW, &WtZ, M_params, &aux_memory, cublas_handle, stream);
    update_w(W, H, X, &Z, &sumH2, &ZHt, N_params, &aux_memory, cublas_handle, stream);
    cudaAssert(cudaStreamEndCapture(stream, &graph));
    cudaAssert(cudaGraphInstantiate(&graph_exec, graph, 0));

    for(uint32_t i = 0; i < max_iter; i++) {
        cudaAssert(cudaGraphLaunch(graph_exec, stream));
    }
}

void update_h(
    Matrix *W, Matrix *H, Matrix *X, Matrix *Z, Matrix *sumW, Matrix *WtZ, uint32_t *M_params, Memory *aux_memory,
    cublasHandle_t cublas_handle, cudaStream_t stream
) {
    uint32_t BLOCK_SIZE = 128;

    // WH = W*H
    matrix_multiply(W, H, Z, cublas_handle);

    // WH = WH+EPS
    Z->set_epsilon(BLOCK_SIZE, stream);

    // Z = X./WH
    element_divide(X, Z, Z, BLOCK_SIZE, stream);

    // sum cols of W into row vector
    W->sum_cols(sumW, aux_memory, M_params, stream);
    sumW->set_epsilon(32, stream);

    // WtZ = W'*Z
    matrix_multiply_AtB(W, Z, WtZ, cublas_handle);

    // WtZ = WtZ./(repmat(sum(W)',1,H.cols)
    //[element divide cols of WtZ by sumW']
    col_divide(WtZ, sumW, WtZ, stream);

    // H = H.*WtZ
    element_multiply(H, WtZ, H, BLOCK_SIZE, stream);
}

void update_w(
    Matrix *W, Matrix *H, Matrix *X, Matrix *Z, Matrix *sumH2, Matrix *ZHt, uint32_t *N_params, Memory *aux_memory,
    cublasHandle_t cublas_handle, cudaStream_t stream
) {
    uint32_t BLOCK_SIZE = 128;

    // WH = W*H
    matrix_multiply(W, H, Z, cublas_handle);

    // WH = WH+EPS
    Z->set_epsilon(BLOCK_SIZE, stream);

    // Z = X./WH
    element_divide(X, Z, Z, BLOCK_SIZE, stream);

    // sum rows of H into col vector
    H->sum_rows(sumH2, aux_memory, N_params, stream);
    sumH2->set_epsilon(32, stream);

    // ZHt = Z*H'
    matrix_multiply_ABt(Z, H, ZHt, cublas_handle);

    // ZHt = ZHt./(repmat(sum(H,2)',W.rows,1)
    //[element divide rows of ZHt by sumH2']
    row_divide(ZHt, sumH2, ZHt, stream);

    // W = W.*ZHt
    element_multiply(W, ZHt, W, BLOCK_SIZE, stream);
}

uint32_t nextpow2(uint32_t x) {
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x + 1;
}

Matrix read_matrix(std::string file, cudaStream_t stream) {
    // read Matrix in from file, store in column-major order

    FILE *fp;
    size_t count;

    uint32_t rows, cols;

    fp = fopen(file.c_str(), "rb");
    count = fread(&rows, sizeof(uint32_t), 1, fp);
    if(count < 1) fprintf(stderr, "read_matrix: fread error\n");
    count = fread(&cols, sizeof(uint32_t), 1, fp);
    if(count < 1) fprintf(stderr, "read_matrix: fread error\n");

    size_t size = rows * cols;
    float *temp = (float *) malloc(sizeof(float) * size);
    count = fread(temp, sizeof(float), size, fp);
    if(count < size) fprintf(stderr, "read_matrix: fread error\n");
    fclose(fp);

    Matrix A(temp, rows, cols, stream);

    // make sure no zero elements
    A.set_epsilon(128, stream);

    free(temp);

    printf("read %s [%ix%i]\n", file.c_str(), A.rows_padded, A.cols_padded);

    return A;
}

void write_matrix(Matrix *matrix, std::string file, cudaStream_t stream) {
    // write Matrix to file using column-major order. Dimensions are written as leading ints

    assert(matrix->rows <= matrix->rows_padded);
    assert(matrix->cols <= matrix->cols_padded);

    float *temp;
    cudaAssert(cudaMallocHost((void **) &temp, matrix->rows * matrix->cols * sizeof(float)));
    cudaAssert(cudaMemcpy2DAsync(
        temp, sizeof(float) * matrix->rows, matrix->data, sizeof(float) * matrix->rows_padded,
        sizeof(float) * matrix->rows, matrix->cols, cudaMemcpyDeviceToHost, stream
    ));
    cudaAssert(cudaStreamSynchronize(stream));

    FILE *fp;
    size_t count;

    fp = fopen(file.c_str(), "wb");

    count = fwrite(&(matrix->rows), sizeof(uint32_t), 1, fp);
    if(count < 1) {
        fprintf(stderr, "write_matrix: fwrite error\n");
    }

    count = fwrite(&(matrix->cols), sizeof(uint32_t), 1, fp);
    if(count < 1) {
        fprintf(stderr, "write_matrix: fwrite error\n");
    }

    count = fwrite(temp, sizeof(float), matrix->rows * matrix->cols, fp);
    if(count < (size_t) (matrix->rows * matrix->cols)) {
        fprintf(stderr, "write_matrix: fwrite error\n");
    }

    fclose(fp);

    cudaAssert(cudaFreeHost(temp));

    printf("write %s [%ix%i]\n", file.c_str(), matrix->rows, matrix->cols);
}
