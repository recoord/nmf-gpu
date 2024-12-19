#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix.cuh"

#define ITER_CHECK 25     // status printed and convergence check every ITER_CHECK iterations
#define MAX_ITER 200      // max number of iterations
#define CONVERGE_THRESH 0 // set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise

void update_div(
    matrix W0, matrix H0, matrix X0, const float thresh, const int32_t max_iter, int32_t verbose,
    cudaStream_t stream
);
uint32_t nextpow2(uint32_t x);


int32_t main(int32_t argc, char *argv[]) {
    cudaStream_t stream = NULL;

    matrix W = read_matrix("../W.bin", stream);
    matrix X = read_matrix("../X.bin", stream);
    matrix H = read_matrix("../H.bin", stream);

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
    matrix W0, matrix H0, matrix X0, const float thresh, const int32_t max_iter, int32_t verbose,
    cudaStream_t stream
) {
    // run iterative multiplicative updates on W,H

    cublasInit();

    const int32_t M = W0.dim[0];
    const int32_t K = W0.dim[1];
    const int32_t N = H0.dim[1];

    // pad matrix dimensions to multiples of:
    const int32_t PAD_MULT = 32;

    int32_t M_padded = M;
    if(M % PAD_MULT != 0) M_padded = M + (PAD_MULT - (M % PAD_MULT));

    int32_t K_padded = K;
    if(K % PAD_MULT != 0) K_padded = K + (PAD_MULT - (K % PAD_MULT));

    int32_t N_padded = N;
    if(N % PAD_MULT != 0) N_padded = N + (PAD_MULT - (N % PAD_MULT));

    // find reduction parameters
    int32_t MN_params[4] = {1, 1, 1, 1}; // M*N size reduction (whole matrix)
    int32_t N_params[4] = {1, 1, 1, 1};  // N size reductions (rows)
    int32_t M_params[4] = {1, 1, 1, 1};  // M size reductions (cols)

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

    MN_params[0] = M_params[0];
    MN_params[1] = M_params[1];
    MN_params[2] = N_params[0];
    MN_params[3] = N_params[1];

    // block size in vector arithmetic operations
    const int32_t BLOCK_SIZE = 128;

    // copy host matrices to device memory
    // copy_matrix_to_device(&W0);
    // copy_matrix_to_device(&H0);
    // copy_matrix_to_device(&X0);

    // matrix to hold W*H
    matrix WH0;
    create_matrix_on_device(&WH0, M, N, 0.0f);

    // compute initial divergence and error
    // float diff, div, change, prev_diff, prev_div;
    matrix_multiply_d(W0, H0, WH0);
    float diff = matrix_difference_norm_d(compute, X0, WH0, MN_params);
    float div = matrix_div_d(compute, X0, WH0, MN_params);

    free_matrix_on_device(&WH0);


    // initialize temp matrices -----------------------

    // matrix to hold X./(W*H+EPS)
    matrix Z;
    create_matrix_on_device(&Z, M_padded, N_padded, 0.0);

    // matrix to hold W'*Z
    matrix WtZ;
    create_matrix_on_device(&WtZ, K_padded, N_padded, 0.0);

    // matrix to hold Z*H'
    matrix ZHt;
    create_matrix_on_device(&ZHt, M_padded, K_padded, 0.0);

    // matrix to hold sum(W) [sum of cols of W]
    matrix sumW;
    create_matrix_on_device(&sumW, 1, K_padded, 0.0);

    // matrix to hold sum(H,2) [sum of rows of H]
    matrix sumH2;
    create_matrix_on_device(&sumH2, K_padded, 1, 0.0);


    // matrices to hold padded versions of matrices
    matrix W;
    create_matrix_on_device(&W, M_padded, K_padded, 0.0);

    matrix H;
    create_matrix_on_device(&H, K_padded, N_padded, 0.0);

    matrix X;
    create_matrix_on_device(&X, M_padded, N_padded, 0.0);


    // move host matrices to padded device memory
    copy_matrix_to_device_padded(W0, W);
    copy_matrix_to_device_padded(H0, H);
    copy_matrix_to_device_padded(X0, X);

    for(int32_t i = 0; i < max_iter; i++) {

        // check for convergence, print status
        if(i % ITER_CHECK == 0 && i != 0) {
            matrix_multiply_d(W, H, Z);
            float prev_diff = diff;
            diff = matrix_difference_norm_d(compute, X, Z, MN_params);
            float change = (prev_diff - diff) / prev_diff;

            if(change < thresh) {
                printf("converged\n");
                break;
            }
        }


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
        sumW.dim[0] = sumW.dim[1];
        sumW.dim[1] = 1;

        // WtZ = W'*Z
        matrix_multiply_AtB_d(W, Z, WtZ);

        // WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
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
        sumH2.dim[1] = sumH2.dim[0];
        sumH2.dim[0] = 1;

        // ZHt = Z*H'
        matrix_multiply_ABt_d(Z, H, ZHt);

        // ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
        //[element divide rows of ZHt by sumH2']
        row_divide_d(ZHt, sumH2, ZHt);

        // W = W.*ZHt
        element_multiply_d(W, ZHt, W, BLOCK_SIZE);

        // reset sumW to row vector
        sumW.dim[1] = sumW.dim[0];
        sumW.dim[0] = 1;
        // reset sumH2 to col vector
        sumH2.dim[0] = sumH2.dim[1];
        sumH2.dim[1] = 1;
    }

    // // reallocate unpadded device memory
    // allocate_matrix_on_device(&W0);
    // allocate_matrix_on_device(&H0);

    // copy padded matrix to unpadded matrices
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

    copy_matrix_to_device(&X0, stream);

    create_matrix_on_device(&WH0, M, N, 0.0);

    // copy device results to host memory
    copy_matrix_from_device(&W0);
    copy_matrix_from_device(&H0);

    // // evaluate final results
    // matrix_multiply_d(W0, H0, WH0);

    // diff = matrix_difference_norm_d(compute, X0, WH0, MN_params);
    // div = matrix_div_d(compute, X0, WH0, MN_params);

    // clean up extra reduction memory
    matrix_difference_norm_d(cleanup, X0, WH0, MN_params);
    matrix_div_d(cleanup, X0, WH0, MN_params);
    sum_cols_d(cleanup, W, sumW, M_params);
    sum_rows_d(cleanup, H, sumH2, N_params);

    // free temp matrices
    destroy_matrix(&WH0);

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
