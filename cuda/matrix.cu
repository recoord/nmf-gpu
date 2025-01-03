#include <cassert>
#include <cuda_runtime_api.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>

#include "error-check.hpp"
#include "matrix.cuh"

#define EPS ((float) (2.2204E-16))
#define MAX_BLOCKS 65535

__global__ void kernel_set_epsilon(float *a, const uint32_t N);
__global__ void vec_div(float *a, float *b, float *c, const uint32_t N);
__global__ void vec_mul(float *a, float *b, float *c, const uint32_t N);
__global__ void col_div(float *a, float *b, float *c, uint32_t M, uint32_t N);
__global__ void col_mul(float *a, float *b, float *c, uint32_t M, uint32_t N);
__global__ void row_div(float *a, float *b, float *c, uint32_t M, uint32_t N);
template <uint32_t blockSize> __global__ void reduce2d(float *g_idata, float *g_odata, uint32_t N);
template <uint32_t blockSize>
__global__ void reduce2d_strided(float *g_idata, float *g_odata, uint32_t N, uint32_t stride);
template <uint32_t blockSize>
__global__ void reduce1d_diff(float *g_idata1, float *g_idata2, float *g_odata, uint32_t N);
template <uint32_t blockSize>
__global__ void reduce1d_div(float *g_idata1, float *g_idata2, float *g_odata, uint32_t N);
void grid2d(dim3 *grid);


Memory::Memory(uint32_t size) {
    cudaAssert(cudaMalloc(&this->data, size));
    this->size = size;
}

Memory::~Memory() {
    if(this->data != nullptr) {
        cudaFree(this->data);
        this->data = nullptr;
        this->size = 0;
    }
}

Matrix::Matrix(uint32_t rows, uint32_t cols) {
    this->rows = rows;
    this->cols = cols;
    this->rows_padded = rows;
    this->cols_padded = cols;

    this->add_padding();

    cudaAssert(cudaMalloc((void **) &(this->data), this->rows_padded * this->cols_padded * sizeof(float)));
}

Matrix::Matrix(float *host_data, uint32_t rows, uint32_t cols, cudaStream_t stream) {
    this->rows = rows;
    this->cols = cols;
    this->rows_padded = rows;
    this->cols_padded = cols;

    this->add_padding();

    uint32_t size = this->rows_padded * this->cols_padded * sizeof(float);
    cudaAssert(cudaMalloc((void **) &(this->data), size));
    cudaAssert(cudaMemcpy2DAsync(
        this->data, this->rows_padded * sizeof(float), host_data, this->rows * sizeof(float),
        this->rows * sizeof(float), this->cols, cudaMemcpyHostToDevice, stream
    ));
}

Matrix::Matrix(float value, uint32_t rows, uint32_t cols, cudaStream_t stream) {
    this->rows = rows;
    this->cols = cols;
    this->rows_padded = rows;
    this->cols_padded = cols;

    this->add_padding();

    uint32_t size = this->rows_padded * this->cols_padded * sizeof(float);
    cudaAssert(cudaMalloc((void **) &(this->data), size));
    cudaAssert(cudaMemsetAsync(this->data, value, size, stream));
}

Matrix::~Matrix() {
    if(this->data != nullptr) {
        cudaFree(this->data);
    }
}

void Matrix::add_padding() {
    if(this->rows != 1 && this->rows % PAD_MULT != 0) {
        this->rows_padded = this->rows + (PAD_MULT - (this->rows % PAD_MULT));
    }
    if(this->cols != 1 && this->cols % PAD_MULT != 0) {
        this->cols_padded = this->cols + (PAD_MULT - (this->cols % PAD_MULT));
    }
}

void matrix_multiply(Matrix *a, Matrix *b, Matrix *c, cublasHandle_t handle) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaAssert(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, c->rows_padded, c->cols_padded, a->cols_padded, &alpha, a->data,
        a->rows_padded, b->data, b->rows_padded, &beta, c->data, c->rows_padded
    ));
}

void matrix_multiply_AtB(Matrix *a, Matrix *b, Matrix *c, cublasHandle_t handle) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaAssert(cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, c->rows_padded, c->cols_padded, b->rows_padded, &alpha, a->data,
        a->rows_padded, b->data, b->rows_padded, &beta, c->data, c->rows_padded
    ));
}

void matrix_multiply_ABt(Matrix *a, Matrix *b, Matrix *c, cublasHandle_t handle) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaAssert(cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, c->rows_padded, c->cols_padded, a->cols_padded, &alpha, a->data,
        a->rows_padded, b->data, b->rows_padded, &beta, c->data, c->rows_padded
    ));
}

void element_divide(Matrix *a, Matrix *b, Matrix *c, uint32_t tile, cudaStream_t stream) {
    // c = a->/b

    if(a->rows_padded != b->rows_padded || a->rows_padded != c->rows_padded || a->cols_padded != b->cols_padded ||
       a->cols_padded != c->cols_padded) {
        fprintf(stderr, "element_divide_d: dimensions do not agree\n");
        exit(1);
    }

    const int32_t N = a->rows_padded * a->cols_padded;
    dim3 block(tile);
    dim3 grid((N / block.x) + (!(N % block.x) ? 0 : 1));
    if(grid.x > MAX_BLOCKS) {
        grid2d(&grid);
    }

    vec_div<<<grid, block, 0, stream>>>(a->data, b->data, c->data, N);
}

__global__ void vec_div(float *a, float *b, float *c, const uint32_t N) {
    const uint32_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        c[i] = a[i] / b[i];
        // c[i] = __fdividef(a[i], b[i]);  // faster, less-accurate divide
    }
}

void element_multiply(Matrix *a, Matrix *b, Matrix *c, uint32_t tile, cudaStream_t stream) {
    // c = a->/b

    if(a->rows_padded != b->rows_padded || a->rows_padded != c->rows_padded || a->cols_padded != b->cols_padded ||
       a->cols_padded != c->cols_padded) {
        fprintf(stderr, "element_multiply_d: dimensions do not agree\n");
        exit(1);
    }

    const int32_t N = a->rows_padded * a->cols_padded;
    dim3 block(tile);
    dim3 grid((N / block.x) + (!(N % block.x) ? 0 : 1));

    if(grid.x > MAX_BLOCKS) {
        grid2d(&grid);
    }

    vec_mul<<<grid, block, 0, stream>>>(a->data, b->data, c->data, N);
}

__global__ void vec_mul(float *a, float *b, float *c, const uint32_t N) {
    const uint32_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N) {
        c[i] = a[i] * b[i];
    }
}

__global__ void kernel_set_epsilon(float *a, const uint32_t N) {
    const uint32_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;

    if(a[i] < EPS && i < N) {
        a[i] = EPS;
    }
}

void Matrix::set_epsilon(uint32_t tile, cudaStream_t stream) {
    uint32_t size = this->rows_padded * this->cols_padded;

    dim3 block(tile);
    dim3 grid((size / block.x) + (!(size % block.x) ? 0 : 1));

    if(grid.x > MAX_BLOCKS) {
        grid2d(&grid);
    }

    kernel_set_epsilon<<<grid, block, 0, stream>>>(this->data, size);
}

void row_divide(Matrix *a, Matrix *b, Matrix *c, cudaStream_t stream) {
    // element divide every row of 'a' by column vector 'b'

    if(a->cols_padded != b->rows_padded || a->rows_padded != c->rows_padded || a->cols_padded != c->cols_padded ||
       b->cols_padded != 1) {
        fprintf(stderr, "row_divide_d: dimension error\n");
        exit(1);
    }

    uint32_t M = a->rows_padded; // number of rows
    uint32_t N = a->cols_padded; // number of cols

    dim3 block(M);
    dim3 grid(N);
    row_div<<<grid, block, 0, stream>>>(a->data, b->data, c->data, M, N);
}

__global__ void row_div(float *a, float *b, float *c, uint32_t M, uint32_t N) {

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] / b[blockIdx.x];
}

void col_divide(Matrix *a, Matrix *b, Matrix *c, cudaStream_t stream) {
    // element divide every column of 'a' by row vector 'b'

    if(a->rows_padded != b->cols_padded || a->rows_padded != c->rows_padded || a->cols_padded != c->cols_padded ||
       b->rows_padded != 1) {
        fprintf(stderr, "col_divide: dimension error\n");
        exit(1);
    }

    uint32_t M = a->rows_padded; // number of rows
    uint32_t N = a->cols_padded; // number of cols
    uint32_t tile = 32;

    dim3 block(tile, 1);
    dim3 grid((M / tile) + (!(M % tile) ? 0 : 1), N);
    col_div<<<grid, block, 0, stream>>>(a->data, b->data, c->data, M, N);
}

__global__ void col_div(float *a, float *b, float *c, uint32_t M, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M) {
        uint32_t ind = i + blockIdx.y * M;
        c[ind] = a[ind] / b[i];
    }
}

__global__ void col_mul(float *a, float *b, float *c, uint32_t M, uint32_t N) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M) {
        uint32_t ind = i + blockIdx.y * M;
        c[ind] = a[ind] * b[i];
    }
}


void Matrix::sum_cols(Matrix *output, Memory *aux_memory, uint32_t *params, cudaStream_t stream) {
    // memory allocated and not freed
    // block1 - block size for first reduction level
    // block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    // lapt1 - load/adds per thread for first red. lev.
    // lapt2 - "" for 2nd ""
    uint32_t block1 = params[0];
    uint32_t tile2 = params[2];
    uint32_t lapt1 = params[1];
    uint32_t lapt2 = params[3];

    if(this->cols_padded != output->cols_padded || output->rows_padded != 1) {
        fprintf(stderr, "sum_cols_d: dimension error\n");
        exit(1);
    }

    const uint32_t N = this->rows_padded; // size of each reduction
    const uint32_t M = this->cols_padded; // number of reductions

    dim3 block(block1, 1);
    dim3 grid((N / (block1 * lapt1)) + (!(N % (block1 * lapt1)) ? 0 : 1), M);

    dim3 block2(tile2, 1);
    dim3 grid2((grid.x / (tile2 * lapt2)) + (!(grid.x % (tile2 * lapt2)) ? 0 : 1), M);

    // Ensure there is enough memory allocated for first level reduction
    assert(aux_memory->size <= grid.x * grid.y * sizeof(float));

    if(tile2 <= 1) { // if we only need one level of reduction
        if(grid.x > 1) {
            fprintf(stderr, "sum_cols_d: grid.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce2d<512><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
            case 256:
                reduce2d<256><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
            case 128:
                reduce2d<128><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
            case 64:
                reduce2d<64><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
            case 32:
                reduce2d<32><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
            case 16:
                reduce2d<16><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
            case 8:
                reduce2d<8><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N);
                break;
        }
    } else { // if we need two levels of reduction
        if(grid2.x > 1) {
            fprintf(stderr, "sum_cols_d: grid2.x > 1\n");
            exit(1);
        }

        switch(block1) {
            case 512:
                reduce2d<512><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
            case 256:
                reduce2d<256><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
            case 128:
                reduce2d<128><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
            case 64:
                reduce2d<64><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
            case 32:
                reduce2d<32><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
            case 16:
                reduce2d<16><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
            case 8:
                reduce2d<8><<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N);
                break;
        }
        switch(tile2) {
            case 512:
                reduce2d<512>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
            case 256:
                reduce2d<256>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
            case 128:
                reduce2d<128>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
            case 64:
                reduce2d<64>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
            case 32:
                reduce2d<32>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
            case 16:
                reduce2d<16>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
            case 8:
                reduce2d<8>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x);
                break;
        }
    }
}

void Matrix::sum_rows(Matrix *output, Memory *aux_memory, uint32_t *params, cudaStream_t stream) {
    // memory allocated and not freed
    // tile1 - block size for first reduction level
    // tile2 - "" for 2nd "" (set to 1 if not using 2nd level)
    // lapt1 - load/adds per thread for first red. lev.
    // lapt2 - "" for 2nd ""

    uint32_t tile1 = params[0];
    uint32_t tile2 = params[2];
    uint32_t lapt1 = params[1];
    uint32_t lapt2 = params[3];

    if(this->rows_padded != output->rows_padded || output->cols_padded != 1) {
        fprintf(stderr, "sum_rows_d: dimension error\n");
        exit(1);
    }

    const uint32_t N = this->cols_padded; // size of each reduction
    const uint32_t M = this->rows_padded; // number of reductions

    dim3 block(tile1, 1);
    dim3 grid((N / (tile1 * lapt1)) + (!(N % (tile1 * lapt1)) ? 0 : 1), M);

    dim3 block2(tile2, 1);
    dim3 grid2((grid.x / (tile2 * lapt2)) + (!(grid.x % (tile2 * lapt2)) ? 0 : 1), M);

    // Ensure there is enough memory allocated for first level reduction
    assert(aux_memory->size <= grid.x * grid.y * sizeof(float));

    if(tile2 <= 1) { // if we only need one level of reduction
        if(grid.x > 1) {
            fprintf(stderr, "sum_rows_d: grid.x > 1\n");
            exit(1);
        }
        switch(tile1) {
            case 512:
                reduce2d_strided<512><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
            case 256:
                reduce2d_strided<256><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
            case 128:
                reduce2d_strided<128><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
            case 64:
                reduce2d_strided<64><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
            case 32:
                reduce2d_strided<32><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
            case 16:
                reduce2d_strided<16><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
            case 8:
                reduce2d_strided<8><<<grid, block, block.x * sizeof(float), stream>>>(this->data, output->data, N, M);
                break;
        }
    } else { // if we need two levels of reduction
        if(grid2.x > 1) {
            fprintf(stderr, "sum_rows_d: grid2.x > 1\n");
            exit(1);
        }

        switch(tile1) {
            case 512:
                reduce2d_strided<512>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
            case 256:
                reduce2d_strided<256>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
            case 128:
                reduce2d_strided<128>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
            case 64:
                reduce2d_strided<64>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
            case 32:
                reduce2d_strided<32>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
            case 16:
                reduce2d_strided<16>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
            case 8:
                reduce2d_strided<8>
                    <<<grid, block, block.x * sizeof(float), stream>>>(this->data, aux_memory->data, N, M);
                break;
        }
        switch(tile2) {
            case 512:
                reduce2d_strided<512>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
            case 256:
                reduce2d_strided<256>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
            case 128:
                reduce2d_strided<128>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
            case 64:
                reduce2d_strided<64>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
            case 32:
                reduce2d_strided<32>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
            case 16:
                reduce2d_strided<16>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
            case 8:
                reduce2d_strided<8>
                    <<<grid2, block2, block2.x * sizeof(float), stream>>>(aux_memory->data, output->data, grid.x, M);
                break;
        }
    }
}

template <uint32_t blockSize>
__global__ void reduce1d_diff(float *g_idata1, float *g_idata2, float *g_odata, uint32_t N) {
    extern __shared__ float sdata[];
    float *diff = (float *) sdata;
    float *sum = (float *) &sdata[blockSize];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockSize + threadIdx.x;
    const uint32_t gridSize = blockSize * gridDim.x;
    sum[tid] = 0;
    diff[tid] = 0;
    while(i < N) {
        diff[tid] += fabs(g_idata1[i] - g_idata2[i]);
        sum[tid] += fabs(g_idata1[i]);
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if(blockSize >= 512) {
        if(tid < 256) {
            diff[tid] += diff[tid + 256];
            sum[tid] += sum[tid + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256) {
        if(tid < 128) {
            diff[tid] += diff[tid + 128];
            sum[tid] += sum[tid + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128) {
        if(tid < 64) {
            diff[tid] += diff[tid + 64];
            sum[tid] += sum[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32) {
        if(blockSize >= 64) {
            diff[tid] += diff[tid + 32];
            sum[tid] += sum[tid + 32];
        }
        if(blockSize >= 32) {
            diff[tid] += diff[tid + 16];
            sum[tid] += sum[tid + 16];
        }
        if(blockSize >= 16) {
            diff[tid] += diff[tid + 8];
            sum[tid] += sum[tid + 8];
        }
        if(blockSize >= 8) {
            diff[tid] += diff[tid + 4];
            sum[tid] += sum[tid + 4];
        }
        if(blockSize >= 4) {
            diff[tid] += diff[tid + 2];
            sum[tid] += sum[tid + 2];
        }
        if(blockSize >= 2) {
            diff[tid] += diff[tid + 1];
            sum[tid] += sum[tid + 1];
        }
    }

    // write result for this block to global mem
    if(tid == 0) {
        g_odata[blockIdx.x + gridDim.x] = sum[0];
        g_odata[blockIdx.x] = diff[0];
    }
}

template <uint32_t blockSize>
__global__ void reduce1d_div(float *g_idata1, float *g_idata2, float *g_odata, uint32_t N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockSize + threadIdx.x;
    const uint32_t gridSize = blockSize * gridDim.x;
    float x;
    float y;
    sdata[tid] = 0;
    while(i < N) {
        x = g_idata1[i];
        y = g_idata2[i];
        // sdata[tid] += (x*__logf(x/y)-x+y);
        sdata[tid] += (x * (logf(x) - logf(y)) - x + y);
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if(blockSize >= 512) {
        if(tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256) {
        if(tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128) {
        if(tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32) {
        if(blockSize >= 64) {
            sdata[tid] += sdata[tid + 32];
        }
        if(blockSize >= 32) {
            sdata[tid] += sdata[tid + 16];
        }
        if(blockSize >= 16) {
            sdata[tid] += sdata[tid + 8];
        }
        if(blockSize >= 8) {
            sdata[tid] += sdata[tid + 4];
        }
        if(blockSize >= 4) {
            sdata[tid] += sdata[tid + 2];
        }
        if(blockSize >= 2) {
            sdata[tid] += sdata[tid + 1];
        }
    }

    // write result for this block to global mem
    if(tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

template <uint32_t blockSize> __global__ void reduce2d(float *g_idata, float *g_odata, uint32_t N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
    const uint32_t offset = blockIdx.y * N;
    const uint32_t gridSize = blockSize * 2 * gridDim.x;
    uint32_t n = N - blockSize;
    sdata[tid] = 0;
    while(i < n) {
        sdata[tid] += g_idata[i + offset] + g_idata[i + offset + blockSize];
        i += gridSize;
    }
    if(i < N) sdata[tid] += g_idata[i + offset];
    __syncthreads();
    // do reduction in shared mem
    if(blockSize >= 512) {
        if(tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256) {
        if(tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128) {
        if(tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32) {
        if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0];
}

template <uint32_t blockSize>
__global__ void reduce2d_strided(float *g_idata, float *g_odata, uint32_t N, uint32_t stride) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
    const uint32_t offset = blockIdx.y;
    const uint32_t gridSize = blockSize * 2 * gridDim.x;
    uint32_t n = N - blockSize;
    sdata[tid] = 0;
    while(i < n) {
        sdata[tid] += g_idata[i * stride + offset] + g_idata[(i + blockSize) * stride + offset];
        i += gridSize;
    }
    if(i < N) sdata[tid] += g_idata[i * stride + offset];
    __syncthreads();
    // do reduction in shared mem
    if(blockSize >= 512) {
        if(tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256) {
        if(tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128) {
        if(tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32) {
        if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if(blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if(blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if(blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if(blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if(blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    // write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.y + blockIdx.x * gridDim.y] = sdata[0];
}

void grid2d(dim3 *grid) {
    // take a 1D grid that is too large and change to 2D
    uint32_t x = grid->x;
    uint32_t y = 1;

    while(x > MAX_BLOCKS) {
        x >>= 2;
        y <<= 2;
    }
    grid->y = y;
    grid->x = grid->x / y + (!(grid->x % y) ? 0 : 1);
}
