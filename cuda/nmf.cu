#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.cuh"

#define ITER_CHECK 25     // status printed and convergence check every ITER_CHECK iterations
#define MAX_ITER 200      // max number of iterations
#define CONVERGE_THRESH 0 // set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise

void update_div(
    Matrix W0, Matrix H0, Matrix X0, const float thresh, const int32_t max_iter, int32_t verbose, cudaStream_t stream
);
uint32_t nextpow2(uint32_t x);


int32_t main(int32_t argc, char *argv[]) {
    cudaStream_t stream = NULL;

    Matrix W = read_matrix("../W.bin", stream);
    Matrix X = read_matrix("../X.bin", stream);
    Matrix H = read_matrix("../H.bin", stream);

    // make sure no zero elements
    matrix_eps_d(X, 128, stream);
    matrix_eps_d(H, 128, stream);
    matrix_eps_d(W, 128, stream);

    // iterative nmf minimization
    update_div(W, H, X, CONVERGE_THRESH, MAX_ITER, 1, stream);

    // write results matrices to binary files
    // (can be read with export_bin.m in Matlab)
    write_matrix(W, "../Wout.bin");
    write_matrix(H, "../Hout.bin");

    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);

    return 0;
}


void update_div(
    Matrix W0, Matrix H0, Matrix X0, const float thresh, const int32_t max_iter, int32_t verbose, cudaStream_t stream
) {
    // run iterative multiplicative updates on W,H

    cublasInit();

    const int32_t M = W0.rows;
    const int32_t K = W0.cols;
    const int32_t N = H0.cols;

    // pad Matrix dimensions to multiples of:
    const int32_t PAD_MULT = 32;

    int32_t M_padded = M;
    if(M % PAD_MULT != 0) M_padded = M + (PAD_MULT - (M % PAD_MULT));

    int32_t K_padded = K;
    if(K % PAD_MULT != 0) K_padded = K + (PAD_MULT - (K % PAD_MULT));

    int32_t N_padded = N;
    if(N % PAD_MULT != 0) N_padded = N + (PAD_MULT - (N % PAD_MULT));

    // find reduction parameters
    int32_t N_params[4] = {1, 1, 1, 1}; // N size reductions (rows)
    int32_t M_params[4] = {1, 1, 1, 1}; // M size reductions (cols)

    int32_t rem;
    rem = nextpow2(N_padded / 128 + (!(N_padded % 128) ? 0 : 1));
    if(rem <= 128) {
        N_params[0] = 128;
        N_params[1] = rem;
    } else if(rem <= 512) {
        N_params[0] = rem;
        N_params[1] = 128;
    } else {
        fprintf(stderr, "reduction parameter error\n");
        exit(1);
    }

    rem = nextpow2(M_padded / 128 + (!(M_padded % 128) ? 0 : 1));
    if(rem <= 128) {
        M_params[0] = 128;
        M_params[1] = rem;
    } else if(rem <= 512) {
        M_params[0] = rem;
        M_params[1] = 128;
    } else {
        fprintf(stderr, "reduction parameter error\n");
        exit(1);
    }

    // block size in vector arithmetic operations
    const int32_t BLOCK_SIZE = 128;

    // initialize temp matrices -----------------------

    // Matrix to hold X./(W*H+EPS)
    Matrix Z(M_padded, N_padded);
    create_matrix_on_device(&Z, M_padded, N_padded, 0.0);

    // Matrix to hold W'*Z
    Matrix WtZ(K_padded, N_padded);
    create_matrix_on_device(&WtZ, K_padded, N_padded, 0.0);

    // Matrix to hold Z*H'
    Matrix ZHt(M_padded, K_padded);
    create_matrix_on_device(&ZHt, M_padded, K_padded, 0.0);

    // Matrix to hold sum(W) [sum of cols of W]
    Matrix sumW(1, K_padded);
    create_matrix_on_device(&sumW, 1, K_padded, 0.0);

    // Matrix to hold sum(H,2) [sum of rows of H]
    Matrix sumH2(K_padded, 1);
    create_matrix_on_device(&sumH2, K_padded, 1, 0.0);


    // matrices to hold padded versions of matrices
    Matrix W(M_padded, K_padded);
    create_matrix_on_device(&W, M_padded, K_padded, 0.0);

    Matrix H(K_padded, N_padded);
    create_matrix_on_device(&H, K_padded, N_padded, 0.0);

    Matrix X(M_padded, N_padded);
    create_matrix_on_device(&X, M_padded, N_padded, 0.0);


    // move host matrices to padded device memory
    copy_matrix_to_device_padded(W0, W);
    copy_matrix_to_device_padded(H0, H);
    copy_matrix_to_device_padded(X0, X);

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

    // copy padded Matrix to unpadded matrices
    copy_from_padded(W0, W);
    copy_from_padded(H0, H);

    // free padded matrices
    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);

    // free temp matrices
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumW);
    destroy_matrix(&sumH2);

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
