#include <cassert>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "error-check.hpp"
#include "matrix.cuh"

#define ITER_CHECK 25     // status printed and convergence check every ITER_CHECK iterations
#define MAX_ITER 200      // max number of iterations
#define CONVERGE_THRESH 0 // set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise

void update_div(
    Matrix W0, Matrix H0, Matrix X0, const uint32_t M, const uint32_t K, const uint32_t N, const float thresh,
    const int32_t max_iter, int32_t verbose, cudaStream_t stream
);
uint32_t nextpow2(uint32_t x);
Matrix read_matrix(std::string file, cudaStream_t stream);
void write_matrix(Matrix A_padded, uint32_t rows, uint32_t cols, std::string file);


int32_t main(int32_t argc, char *argv[]) {
    cudaStream_t stream = NULL;

    Matrix W0 = read_matrix("../W.bin", stream);
    Matrix X0 = read_matrix("../X.bin", stream);
    Matrix H0 = read_matrix("../H.bin", stream);

    // make sure no zero elements
    matrix_eps_d(X0, 128, stream);
    matrix_eps_d(H0, 128, stream);
    matrix_eps_d(W0, 128, stream);

    const uint32_t M = W0.rows;
    const uint32_t K = W0.cols;
    const uint32_t N = H0.cols;

    // matrices to hold padded versions of matrices
    Matrix W(0.0f, M, K, true);
    Matrix H(0.0f, K, N, true);
    Matrix X(0.0f, M, N, true);

    W0.copy_to_padded(&W);
    H0.copy_to_padded(&H);
    X0.copy_to_padded(&X);

    // iterative nmf minimization
    update_div(W, H, X, M, K, N, CONVERGE_THRESH, MAX_ITER, 1, stream);

    // write results matrices to binary files
    // (can be read with export_bin.m in Matlab)
    write_matrix(W, W0.rows, W0.cols, "../Wout.bin");
    write_matrix(H, H0.rows, H0.cols, "../Hout.bin");

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

void update_div(
    Matrix W, Matrix H, Matrix X, const uint32_t M, const uint32_t K, const uint32_t N, const float thresh,
    const int32_t max_iter, int32_t verbose, cudaStream_t stream
) {
    // run iterative multiplicative updates on W,H

    cublasInit();

    // find reduction parameters
    uint32_t N_params[4]; // N size reductions (rows)
    uint32_t M_params[4]; // M size reductions (cols)

    init_params(N, N_params);
    init_params(M, M_params);

    // block size in vector arithmetic operations
    const int32_t BLOCK_SIZE = 128;

    // initialize temp matrices -----------------------
    Matrix Z(0.0f, M, N, true);     // Matrix to hold X./(W*H+EPS)
    Matrix WtZ(0.0f, K, N, true);   // Matrix to hold W'*Z
    Matrix ZHt(0.0f, M, K, true);   // Matrix to hold Z*H'
    Matrix sumW(0.0f, 1, K, true);  // Matrix to hold sum(W) [sum of cols of W]
    Matrix sumH2(0.0f, K, 1, true); // Matrix to hold sum(H,2) [sum of rows of H]

    for(int32_t i = 0; i < max_iter; i++) {
        /* matlab algorithm
           Z = X./(W*H+eps); H = H.*(W'*Z)./(repmat(sum(W)',1,F));
           Z = X./(W*H+eps);
           W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
           */

        //
        // UPDATE H -----------------------------
        //

        // WH = W*H
        matrix_multiply_d(W, H, Z);

        // WH = WH+EPS
        matrix_eps_d(Z, BLOCK_SIZE, stream);

        // Z = X./WH
        element_divide_d(X, Z, Z, BLOCK_SIZE);

        // sum cols of W into row vector
        sum_cols_d(compute, W, sumW, M_params);
        matrix_eps_d(sumW, 32, stream);

        // convert sumW to col vector (transpose)
        sumW.rows = sumW.cols;
        sumW.cols = 1;

        // WtZ = W'*Z
        matrix_multiply_AtB_d(W, Z, WtZ);

        // WtZ = WtZ./(repmat(sum(W)',1,H.cols)
        //[element divide cols of WtZ by sumW']
        col_divide_d(WtZ, sumW, WtZ);

        // H = H.*WtZ
        element_multiply_d(H, WtZ, H, BLOCK_SIZE);

        //
        // UPDATE W ---------------------------
        //

        // WH = W*H
        matrix_multiply_d(W, H, Z);

        // WH = WH+EPS
        matrix_eps_d(Z, BLOCK_SIZE, stream);

        // Z = X./WH
        element_divide_d(X, Z, Z, BLOCK_SIZE);

        // sum rows of H into col vector
        sum_rows_d(compute, H, sumH2, N_params);
        matrix_eps_d(sumH2, 32, stream);

        // convert sumH2 to row vector (transpose)
        sumH2.cols = sumH2.rows;
        sumH2.rows = 1;

        // ZHt = Z*H'
        matrix_multiply_ABt_d(Z, H, ZHt);

        // ZHt = ZHt./(repmat(sum(H,2)',W.rows,1)
        //[element divide rows of ZHt by sumH2']
        row_divide_d(ZHt, sumH2, ZHt);

        // W = W.*ZHt
        element_multiply_d(W, ZHt, W, BLOCK_SIZE);

        // reset sumW to row vector
        sumW.cols = sumW.rows;
        sumW.rows = 1;
        // reset sumH2 to col vector
        sumH2.rows = sumH2.cols;
        sumH2.cols = 1;
    }

    // clean up extra reduction memory
    sum_cols_d(cleanup, W, sumW, M_params);
    sum_rows_d(cleanup, H, sumH2, N_params);

    cublasShutdown();
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

    Matrix A(temp, rows, cols, false);

    free(temp);

    printf("read %s [%ix%i]\n", file.c_str(), A.rows, A.cols);

    return A;
}

void write_matrix(Matrix A_padded, uint32_t rows, uint32_t cols, std::string file) {
    // write Matrix to file using column-major order
    // dimensions are written as leading ints

    assert(rows <= A_padded.rows);
    assert(cols <= A_padded.cols);

    float *temp;
    cudaAssert(cudaMallocHost((void **) &temp, rows * cols * sizeof(float)));
    cudaMemcpy2D(
        temp, sizeof(float) * rows, A_padded.data, sizeof(float) * A_padded.rows, sizeof(float) * rows, cols,
        cudaMemcpyDeviceToHost
    );

    FILE *fp;
    size_t count;

    fp = fopen(file.c_str(), "wb");
    count = fwrite(&(rows), sizeof(uint32_t), 1, fp);
    if(count < 1) fprintf(stderr, "write_matrix: fwrite error\n");
    count = fwrite(&(cols), sizeof(uint32_t), 1, fp);
    if(count < 1) fprintf(stderr, "write_matrix: fwrite error\n");

    count = fwrite(temp, sizeof(float), rows * cols, fp);
    if(count < (size_t) (rows * cols)) fprintf(stderr, "write_matrix: fwrite error\n");
    fclose(fp);

    cudaAssert(cudaFreeHost(temp));

    printf("write %s [%ix%i]\n", file.c_str(), rows, cols);
}
