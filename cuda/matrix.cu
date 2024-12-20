#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "error-check.hpp"
#include "matrix.cuh"

#define EPS (float) (2.2204E-16)
#define MAX_BLOCKS 65535


__global__ void vecEps(float *a, const int32_t N);
__global__ void vecDiv(float *a, float *b, float *c, const int32_t N);
__global__ void vecMult(float *a, float *b, float *c, const int32_t N);
__global__ void colDiv(float *a, float *b, float *c, int32_t M, int32_t N);
__global__ void colMul(float *a, float *b, float *c, int32_t M, int32_t N);
__global__ void rowDiv(float *a, float *b, float *c, int32_t M, int32_t N);
template <uint32_t blockSize> __global__ void reduce2D(float *g_idata, float *g_odata, int32_t N);
template <uint32_t blockSize>
__global__ void reduce2DStrided(float *g_idata, float *g_odata, int32_t N, int32_t stride);
template <uint32_t blockSize> __global__ void reduce1DDiff(float *g_idata1, float *g_idata2, float *g_odata, int32_t N);
template <uint32_t blockSize> __global__ void reduce1DDiv(float *g_idata1, float *g_idata2, float *g_odata, int32_t N);
template <uint32_t blockSize> __global__ void reduce1DNan(float *g_idata1, float *g_odata, int32_t N);
template <uint32_t blockSize> __global__ void reduce1DEql(float *g_idata1, float *g_odata, int32_t N);
void grid2D(dim3 *dimGrid);

Matrix::Matrix(uint32_t rows, uint32_t cols) {
    this->rows = rows;
    this->cols = cols;
    this->data = nullptr;
}

Matrix::~Matrix() {}


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

    Matrix A(rows, cols);

    size_t N = A.rows * A.cols;
    float *temp = (float *) malloc(sizeof(float) * N);
    count = fread(temp, sizeof(float), N, fp);
    if(count < N) fprintf(stderr, "read_matrix: fread error\n");
    fclose(fp);

    // copy_matrix_to_device(&A, stream);
    cudaAssert(cudaMalloc((void **) &(A.data), N * sizeof(float)));
    cudaAssert(cudaMemcpyAsync(A.data, temp, N * sizeof(float), cudaMemcpyHostToDevice, stream));

    free(temp);

    printf("read %s [%ix%i]\n", file.c_str(), A.rows, A.cols);

    return A;
}

void write_matrix(Matrix A, std::string file) {
    // write Matrix to file using column-major order
    // dimensions are written as leading ints

    size_t size = A.rows * A.cols * sizeof(float);
    float *temp;
    cudaAssert(cudaMallocHost((void **) &temp, size));
    cudaAssert(cudaMemcpy(temp, A.data, size, cudaMemcpyDeviceToHost));

    FILE *fp;
    size_t count;

    fp = fopen(file.c_str(), "wb");
    count = fwrite(&(A.rows), sizeof(uint32_t), 1, fp);
    if(count < 1) fprintf(stderr, "write_matrix: fwrite error\n");
    count = fwrite(&(A.cols), sizeof(uint32_t), 1, fp);
    if(count < 1) fprintf(stderr, "write_matrix: fwrite error\n");

    count = fwrite(temp, sizeof(float), A.rows * A.cols, fp);
    if(count < (size_t) (A.rows * A.cols)) fprintf(stderr, "write_matrix: fwrite error\n");
    fclose(fp);

    cudaAssert(cudaFreeHost(temp));

    printf("write %s [%ix%i]\n", file.c_str(), A.rows, A.cols);
}

void create_matrix_on_device(Matrix *A, int32_t rows, int32_t cols, float value) {
    // create Matrix on device  with all elements equal to 'value'
    // Matrix dimensions are in dim[] {rows,cols}

    A->rows = rows;
    A->cols = cols;
    // A->mat = NULL;

    const int32_t N = A->rows * A->cols;

    cudaError_t err;
    err = cudaMalloc((void **) &(A->data), sizeof(float) * N);
    // printf("device pointer: %p\n",A->data);
    if(err != cudaSuccess) {
        fprintf(stderr, "create_matrix_on_device: cudaMalloc: ErrorMemoryAllocation\n");
        exit(1);
    }

    float *temp = (float *) malloc(sizeof(float) * N);
    for(int32_t i = 0; i < N; i++) temp[i] = value;
    cudaMemcpy(A->data, temp, sizeof(float) * N, cudaMemcpyHostToDevice);

    free(temp);
}

/*
void copy_to_padded_with_cols(Matrix A, Matrix Apad){
    //create Matrix on device  with all elements equal to 'value'
    //Matrix dimensions are in dim[] {rows,cols}

    const int32_t M = A.rows;
    const int32_t N = A.cols;
    const int32_t M_padded = Apad.rows;
    const int32_t N_padded = Apad.cols;

    if (M != M_padded){
    fprintf(stderr,"copy_to_padded_with_cols: number of rows must stay the same\n");
    exit(1);
    }
    if (N > N_padded){
    fprintf(stderr,"copy_to_padded_with_cols: padded number of cols must be >= original\n");
    exit(1);
    }

    cudaMemcpy(Apad.data,A.data,sizeof(float)*N*M,cudaMemcpyDeviceToDevice);




}
*/

void copy_to_padded(Matrix A, Matrix Apad) {
    // copy unpadded Matrix on device to padded Matrix on device

    const int32_t M = A.rows;
    const int32_t N = A.cols;
    const int32_t M_padded = Apad.rows;
    const int32_t N_padded = Apad.cols;

    if(M > M_padded) {
        fprintf(stderr, "copy_to_padded: padded number of rows must be >= original\n");
        exit(1);
    }
    if(N > N_padded) {
        fprintf(stderr, "copy_to_padded: padded number of cols must be >= original\n");
        exit(1);
    }

    cudaError_t err;
    err = cudaMemcpy2D(
        Apad.data, sizeof(float) * M_padded, A.data, sizeof(float) * M, sizeof(float) * M, N, cudaMemcpyDeviceToDevice
    );
    if(err != cudaSuccess) {
        fprintf(stderr, "copy_to_padded: error in cudaMemcpy2D [%i],%i\n", err, cudaErrorInvalidValue);
        exit(1);
    }
    // cudaMemcpy2D( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum
    // cudaMemcpyKind kind )
}

void copy_matrix_to_device_padded(Matrix A, Matrix Apad) {
    // copy unpadded Matrix on host to padded Matrix on device

    const int32_t M = A.rows;
    const int32_t N = A.cols;
    const int32_t M_padded = Apad.rows;
    const int32_t N_padded = Apad.cols;

    if(M > M_padded) {
        fprintf(stderr, "copy_to_padded: padded number of rows must be >= original\n");
        exit(1);
    }
    if(N > N_padded) {
        fprintf(stderr, "copy_to_padded: padded number of cols must be >= original\n");
        exit(1);
    }

    cudaError_t err;
    err = cudaMemcpy2D(
        Apad.data, sizeof(float) * M_padded, A.data, sizeof(float) * M, sizeof(float) * M, N, cudaMemcpyDeviceToDevice
    );
    if(err != cudaSuccess) {
        fprintf(stderr, "copy_to_padded: error in cudaMemcpy2D [%i],%i\n", err, cudaErrorInvalidValue);
        exit(1);
    }
    // cudaMemcpy2D( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum
    // cudaMemcpyKind kind )
}

void copy_from_padded(Matrix A, Matrix Apad) {
    // copy padded Matrix on device to unpadded Matrix on device

    const int32_t M = A.rows;
    const int32_t N = A.cols;
    const int32_t M_padded = Apad.rows;
    const int32_t N_padded = Apad.cols;

    if(M > M_padded) {
        fprintf(stderr, "copy_from_padded: padded number of rows must be >= original\n");
        exit(1);
    }
    if(N > N_padded) {
        fprintf(stderr, "copy_from_padded: padded number of cols must be >= original\n");
        exit(1);
    }

    cudaMemcpy2D(
        A.data, sizeof(float) * M, Apad.data, sizeof(float) * M_padded, sizeof(float) * M, N, cudaMemcpyDeviceToDevice
    );
}

void destroy_matrix(Matrix *A) {
    if(A->data != NULL) cudaFree(A->data);
    A->data = NULL;

    A->rows = 0;
    A->cols = 0;
}

void free_matrix_on_device(Matrix *A) {
    if(A->data != NULL) cudaFree(A->data);
    A->data = NULL;
}

void allocate_matrix_on_device(Matrix *A) {

    const int32_t N = A->rows * A->cols;
    cudaError_t err;

    if(A->data == NULL) {
        err = cudaMalloc((void **) &(A->data), sizeof(float) * N);
        if(err != cudaSuccess) {
            fprintf(stderr, "allocate_matrix_on_device: cudaMalloc: FAIL\n");
            exit(1);
        }
    } else {
        fprintf(stderr, "allocate_matrix_on_device: Matrix already allocated on device");
        exit(1);
    }
}

void copy_matrix_on_device(Matrix A, Matrix B) {

    if(A.rows != B.rows || A.cols != B.cols) {
        fprintf(stderr, "copy_matrix_on_device: dimension error\n");
        exit(1);
    }
    const int32_t N = A.rows * A.cols;

    if(A.data == NULL) {
        fprintf(stderr, "copy_matrix_on_device: source Matrix not allocated on device\n");
        exit(1);
    }
    if(B.data == NULL) {
        fprintf(stderr, "copy_matrix_on_device: dest. Matrix not allocated on device\n");
        exit(1);
    }

    cudaAssert(cudaMemcpy(B.data, A.data, sizeof(float) * N, cudaMemcpyDeviceToDevice));
}

void matrix_multiply_d(Matrix a, Matrix b, Matrix c) {
    // TODO: Is this the legacy API?
    cublasSgemm('N', 'N', c.rows, c.cols, a.cols, 1, a.data, a.rows, b.data, b.rows, 0, c.data, c.rows);
    cudaAssert(cublasGetError());
}

void matrix_multiply_AtB_d(Matrix a, Matrix b, Matrix c) {
    // TODO: Is this the legacy API?
    cublasSgemm('T', 'N', c.rows, c.cols, b.rows, 1, a.data, a.rows, b.data, b.rows, 0, c.data, c.rows);
    cudaAssert(cublasGetError());
}

void matrix_multiply_ABt_d(Matrix a, Matrix b, Matrix c) {
    // TODO: Is this the legacy API?
    cublasSgemm('N', 'T', c.rows, c.cols, a.cols, 1, a.data, a.rows, b.data, b.rows, 0, c.data, c.rows);
    cudaAssert(cublasGetError());
}

void element_divide_d(Matrix a, Matrix b, Matrix c, int32_t block_size) {
    // c = a./b

    if(a.rows != b.rows || a.rows != c.rows || a.cols != b.cols || a.cols != c.cols) {
        fprintf(stderr, "element_divide_d: dimensions do not agree\n");
        exit(1);
    }

    const int32_t N = a.rows * a.cols;
    dim3 dimBlock(block_size);
    dim3 dimGrid((N / dimBlock.x) + (!(N % dimBlock.x) ? 0 : 1));
    if(dimGrid.x > MAX_BLOCKS) grid2D(&dimGrid);

    vecDiv<<<dimGrid, dimBlock>>>(a.data, b.data, c.data, N);
}

__global__ void vecDiv(float *a, float *b, float *c, const int32_t N) {
    // const int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    const int32_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) c[i] = a[i] / b[i];
    // c[i] = __fdividef(a[i],b[i]);  //faster, less-accurate divide
}

void element_multiply_d(Matrix a, Matrix b, Matrix c, int32_t block_size) {
    // c = a./b

    if(a.rows != b.rows || a.rows != c.rows || a.cols != b.cols || a.cols != c.cols) {
        fprintf(stderr, "element_multiply_d: dimensions do not agree\n");
        exit(1);
    }

    const int32_t N = a.rows * a.cols;
    dim3 dimBlock(block_size);
    dim3 dimGrid((N / dimBlock.x) + (!(N % dimBlock.x) ? 0 : 1));

    if(dimGrid.x > MAX_BLOCKS) grid2D(&dimGrid);

    vecMult<<<dimGrid, dimBlock>>>(a.data, b.data, c.data, N);
}

__global__ void vecMult(float *a, float *b, float *c, const int32_t N) {
    // const int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    const int32_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) c[i] = a[i] * b[i];
}

__global__ void vecEps(float *a, const int32_t N) {
    const int32_t i = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;

    if(a[i] < EPS && i < N) a[i] = EPS;
}

void matrix_eps_d(Matrix a, int32_t block_size, cudaStream_t stream) {

    const int32_t N = a.rows * a.cols;

    dim3 dimBlock(block_size);
    dim3 dimGrid((N / dimBlock.x) + (!(N % dimBlock.x) ? 0 : 1));

    if(dimGrid.x > MAX_BLOCKS) grid2D(&dimGrid);

    vecEps<<<dimGrid, dimBlock, 0, stream>>>(a.data, N);
}

void row_divide_d(Matrix a, Matrix b, Matrix c) {
    // element divide every row of 'a' by row vector 'b'

    if(a.cols != b.cols || a.rows != c.rows || a.cols != c.cols || b.rows != 1) {
        fprintf(stderr, "row_divide_d: dimension error\n");
        exit(1);
    }
    int32_t M = a.rows; // number of rows
    int32_t N = a.cols; // number of cols

    dim3 dimBlock(M);
    dim3 dimGrid(N);
    rowDiv<<<dimGrid, dimBlock>>>(a.data, b.data, c.data, M, N);
}

__global__ void rowDiv(float *a, float *b, float *c, int32_t M, int32_t N) {

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] / b[blockIdx.x];
}

void col_divide_d(Matrix a, Matrix b, Matrix c) {
    // element divide every column of 'a' by column vector 'b'

    if(a.rows != b.rows || a.rows != c.rows || a.cols != c.cols || b.cols != 1) {
        fprintf(stderr, "col_divide: dimension error\n");
        exit(1);
    }
    int32_t M = a.rows; // number of rows
    int32_t N = a.cols; // number of cols
    int32_t block = 32;

    dim3 dimBlock(block, 1);
    dim3 dimGrid((M / block) + (!(M % block) ? 0 : 1), N);
    colDiv<<<dimGrid, dimBlock>>>(a.data, b.data, c.data, M, N);
}

__global__ void colDiv(float *a, float *b, float *c, int32_t M, int32_t N) {

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M) {
        int32_t ind = i + blockIdx.y * M;
        c[ind] = a[ind] / b[i];
    }
}

__global__ void colMul(float *a, float *b, float *c, int32_t M, int32_t N) {

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M) {
        int32_t ind = i + blockIdx.y * M;
        c[ind] = a[ind] * b[i];
    }
}

void sum_cols_d(action_t action, Matrix a, Matrix c, int32_t *params) {
    // memory allocated and not freed
    // block1 - block size for first reduction level
    // block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    // lapt1 - load/adds per thread for first red. lev.
    // lapt2 - "" for 2nd ""
    int32_t block1 = params[0];
    int32_t block2 = params[2];
    int32_t lapt1 = params[1];
    int32_t lapt2 = params[3];

    static uint32_t r1size = 0;
    static float *r1 = NULL;
    if(action == cleanup) {
        if(r1 != NULL) {
            cudaFree(r1);
            r1 = NULL;
        }
        r1size = 0;
        return;
    }

    if(a.cols != c.cols || c.rows != 1) {
        fprintf(stderr, "sum_cols_d: dimension error\n");
        exit(1);
    }

    const int32_t N = a.rows; // size of each reduction
    const int32_t M = a.cols; // number of reductions

    dim3 dimBlock(block1, 1);
    dim3 dimGrid((N / (block1 * lapt1)) + (!(N % (block1 * lapt1)) ? 0 : 1), M);

    dim3 dimBlock2(block2, 1);
    dim3 dimGrid2((dimGrid.x / (block2 * lapt2)) + (!(dimGrid.x % (block2 * lapt2)) ? 0 : 1), M);

    // printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    // printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    // allocate memory for first level reduction
    if(r1size < dimGrid.x * dimGrid.y) {
        if(r1 != NULL) cudaFree(r1);
        r1size = dimGrid.x * dimGrid.y;
        cudaMalloc((void **) &r1, sizeof(float) * r1size);
    }

    if(block2 <= 1) { // if we only need one level of reduction
        if(dimGrid.x > 1) {
            fprintf(stderr, "sum_cols_d: dimGrid.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce2D<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
            case 256:
                reduce2D<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
            case 128:
                reduce2D<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
            case 64:
                reduce2D<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
            case 32:
                reduce2D<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
            case 16:
                reduce2D<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
            case 8:
                reduce2D<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N);
                break;
        }
    } else { // if we need two levels of reduction
        if(dimGrid2.x > 1) {
            fprintf(stderr, "sum_cols_d: dimGrid2.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce2D<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 256:
                reduce2D<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 128:
                reduce2D<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 64:
                reduce2D<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 32:
                reduce2D<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 16:
                reduce2D<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 8:
                reduce2D<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
        }
        switch(block2) {
            case 512:
                reduce2D<512><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
            case 256:
                reduce2D<256><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
            case 128:
                reduce2D<128><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
            case 64:
                reduce2D<64><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
            case 32:
                reduce2D<32><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
            case 16:
                reduce2D<16><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
            case 8:
                reduce2D<8><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x);
                break;
        }
    }
}

void sum_rows_d(action_t action, Matrix a, Matrix c, int32_t *params) {
    // memory allocated and not freed
    // block1 - block size for first reduction level
    // block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    // lapt1 - load/adds per thread for first red. lev.
    // lapt2 - "" for 2nd ""

    int32_t block1 = params[0];
    int32_t block2 = params[2];
    int32_t lapt1 = params[1];
    int32_t lapt2 = params[3];

    static uint32_t r1size = 0;
    static float *r1 = NULL;
    if(action == cleanup) {
        if(r1 != NULL) {
            cudaFree(r1);
            r1 = NULL;
        }
        r1size = 0;
        return;
    }
    if(a.rows != c.rows || c.cols != 1) {
        fprintf(stderr, "sum_rows_d: dimension error\n");
        exit(1);
    }

    const int32_t N = a.cols; // size of each reduction
    const int32_t M = a.rows; // number of reductions

    dim3 dimBlock(block1, 1);
    dim3 dimGrid((N / (block1 * lapt1)) + (!(N % (block1 * lapt1)) ? 0 : 1), M);

    dim3 dimBlock2(block2, 1);
    dim3 dimGrid2((dimGrid.x / (block2 * lapt2)) + (!(dimGrid.x % (block2 * lapt2)) ? 0 : 1), M);

    // printf("1: %i %i %i %i\n",dimBlock.x,dimBlock.y, dimGrid.x, dimGrid.y);
    // printf("2: %i %i %i %i\n",dimBlock2.x,dimBlock2.y, dimGrid2.x, dimGrid2.y);

    // allocate memory for first level reduction
    if(r1size < dimGrid.x * dimGrid.y) {
        if(r1 != NULL) cudaFree(r1);
        r1size = dimGrid.x * dimGrid.y;
        cudaMalloc((void **) &r1, sizeof(float) * r1size);
    }

    if(block2 <= 1) { // if we only need one level of reduction
        if(dimGrid.x > 1) {
            fprintf(stderr, "sum_rows_d: dimGrid.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce2DStrided<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
            case 256:
                reduce2DStrided<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
            case 128:
                reduce2DStrided<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
            case 64:
                reduce2DStrided<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
            case 32:
                reduce2DStrided<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
            case 16:
                reduce2DStrided<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
            case 8:
                reduce2DStrided<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, c.data, N, M);
                break;
        }
    } else { // if we need two levels of reduction
        if(dimGrid2.x > 1) {
            fprintf(stderr, "sum_rows_d: dimGrid2.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce2DStrided<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
            case 256:
                reduce2DStrided<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
            case 128:
                reduce2DStrided<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
            case 64:
                reduce2DStrided<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
            case 32:
                reduce2DStrided<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
            case 16:
                reduce2DStrided<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
            case 8:
                reduce2DStrided<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N, M);
                break;
        }
        switch(block2) {
            case 512:
                reduce2DStrided<512><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
            case 256:
                reduce2DStrided<256><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
            case 128:
                reduce2DStrided<128><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
            case 64:
                reduce2DStrided<64><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
            case 32:
                reduce2DStrided<32><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
            case 16:
                reduce2DStrided<16><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
            case 8:
                reduce2DStrided<8><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, c.data, dimGrid.x, M);
                break;
        }
    }
}

float nan_check_d(action_t action, Matrix a, int32_t *params) {
    // memory allocated and not freed
    // block1 - block size for first reduction level
    // block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    // lapt1 - load/adds per thread for first red. lev.
    // lapt2 - "" for 2nd ""

    int32_t block1 = params[0];
    int32_t block2 = params[2];
    int32_t lapt1 = params[1];
    int32_t lapt2 = params[3];

    static uint32_t r1size = 0;
    static float *r1 = NULL;
    static float *result_d = NULL;
    if(action == cleanup) {
        if(r1 != NULL) {
            cudaFree(r1);
            r1 = NULL;
        }
        if(result_d != NULL) {
            cudaFree(result_d);
            result_d = NULL;
        }
        r1size = 0;
        return 0;
    }


    const int32_t N = a.rows * a.cols; // size of each reduction

    dim3 dimBlock(block1);
    dim3 dimGrid((N / (block1 * lapt1)) + (!(N % (block1 * lapt1)) ? 0 : 1));

    dim3 dimBlock2(block2);
    dim3 dimGrid2((dimGrid.x / (block2 * lapt2)) + (!(dimGrid.x % (block2 * lapt2)) ? 0 : 1));

    // allocate memory for first level reduction
    if(result_d == NULL) cudaMalloc((void **) &result_d, sizeof(float) * 1);
    if(r1size < dimGrid.x) {
        if(r1 != NULL) cudaFree(r1);
        r1size = dimGrid.x;
        cudaMalloc((void **) &r1, sizeof(float) * r1size);
    }

    if(block2 <= 1) { // if we only need one level of reduction
        if(dimGrid.x > 1) {
            fprintf(stderr, "matrix_difference_norm_d: dimGrid.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce1DNan<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 256:
                reduce1DNan<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 128:
                reduce1DNan<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 64:
                reduce1DNan<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 32:
                reduce1DNan<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 16:
                reduce1DNan<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 8:
                reduce1DNan<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
        }
    } else { // if we need two levels of reduction
        if(dimGrid2.x > 1) {
            fprintf(stderr, "matrix_difference_norm_d: dimGrid2.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce1DNan<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 256:
                reduce1DNan<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 128:
                reduce1DNan<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 64:
                reduce1DNan<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 32:
                reduce1DNan<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 16:
                reduce1DNan<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 8:
                reduce1DNan<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
        }
        switch(block2) {
            case 512:
                reduce2D<512><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 256:
                reduce2D<256><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 128:
                reduce2D<128><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 64:
                reduce2D<64><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 32:
                reduce2D<32><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 16:
                reduce2D<16><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 8:
                reduce2D<8><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
        }
    }

    float result;
    cudaMemcpy(&result, result_d, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

float zero_check_d(action_t action, Matrix a, int32_t *params) {
    // memory allocated and not freed
    // block1 - block size for first reduction level
    // block2 - "" for 2nd "" (set to 1 if not using 2nd level)
    // lapt1 - load/adds per thread for first red. lev.
    // lapt2 - "" for 2nd ""

    int32_t block1 = params[0];
    int32_t block2 = params[2];
    int32_t lapt1 = params[1];
    int32_t lapt2 = params[3];

    static uint32_t r1size = 0;
    static float *r1 = NULL;
    static float *result_d = NULL;
    if(action == cleanup) {
        if(r1 != NULL) {
            cudaFree(r1);
            r1 = NULL;
        }
        if(result_d != NULL) {
            cudaFree(result_d);
            result_d = NULL;
        }
        r1size = 0;
        return 0;
    }


    const int32_t N = a.rows * a.cols; // size of each reduction

    dim3 dimBlock(block1);
    dim3 dimGrid((N / (block1 * lapt1)) + (!(N % (block1 * lapt1)) ? 0 : 1));

    dim3 dimBlock2(block2);
    dim3 dimGrid2((dimGrid.x / (block2 * lapt2)) + (!(dimGrid.x % (block2 * lapt2)) ? 0 : 1));

    // allocate memory for first level reduction
    if(result_d == NULL) cudaMalloc((void **) &result_d, sizeof(float) * 1);
    if(r1size < dimGrid.x) {
        if(r1 != NULL) cudaFree(r1);
        r1size = dimGrid.x;
        cudaMalloc((void **) &r1, sizeof(float) * r1size);
    }

    if(block2 <= 1) { // if we only need one level of reduction
        if(dimGrid.x > 1) {
            fprintf(stderr, "matrix_difference_norm_d: dimGrid.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce1DEql<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 256:
                reduce1DEql<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 128:
                reduce1DEql<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 64:
                reduce1DEql<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 32:
                reduce1DEql<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 16:
                reduce1DEql<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
            case 8:
                reduce1DEql<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, result_d, N);
                break;
        }
    } else { // if we need two levels of reduction
        if(dimGrid2.x > 1) {
            fprintf(stderr, "matrix_difference_norm_d: dimGrid2.x > 1\n");
            exit(1);
        }
        switch(block1) {
            case 512:
                reduce1DEql<512><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 256:
                reduce1DEql<256><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 128:
                reduce1DEql<128><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 64:
                reduce1DEql<64><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 32:
                reduce1DEql<32><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 16:
                reduce1DEql<16><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
            case 8:
                reduce1DEql<8><<<dimGrid, dimBlock, dimBlock.x * sizeof(float)>>>(a.data, r1, N);
                break;
        }
        switch(block2) {
            case 512:
                reduce2D<512><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 256:
                reduce2D<256><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 128:
                reduce2D<128><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 64:
                reduce2D<64><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 32:
                reduce2D<32><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 16:
                reduce2D<16><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
            case 8:
                reduce2D<8><<<dimGrid2, dimBlock2, dimBlock2.x * sizeof(float)>>>(r1, result_d, dimGrid.x);
                break;
        }
    }

    float result;
    cudaMemcpy(&result, result_d, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

template <uint32_t blockSize> __global__ void reduce1DNan(float *g_idata1, float *g_odata, int32_t N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockSize + threadIdx.x;
    const int32_t gridSize = blockSize * gridDim.x;
    float x;
    sdata[tid] = 0;
    while(i < N) {
        x = g_idata1[i];
        // sdata[tid] += (x*__logf(x/y)-x+y);
        sdata[tid] += (float) isnan(x);
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

template <uint32_t blockSize> __global__ void reduce1DEql(float *g_idata1, float *g_odata, int32_t N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockSize + threadIdx.x;
    const int32_t gridSize = blockSize * gridDim.x;
    float x;
    sdata[tid] = 0;
    while(i < N) {
        x = g_idata1[i];
        // sdata[tid] += (float)isinf(x);
        sdata[tid] += (float) (x == 0);
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

template <uint32_t blockSize>
__global__ void reduce1DDiff(float *g_idata1, float *g_idata2, float *g_odata, int32_t N) {
    extern __shared__ float sdata[];
    float *diff = (float *) sdata;
    float *sum = (float *) &sdata[blockSize];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockSize + threadIdx.x;
    const int32_t gridSize = blockSize * gridDim.x;
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

template <uint32_t blockSize> __global__ void reduce1DDiv(float *g_idata1, float *g_idata2, float *g_odata, int32_t N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockSize + threadIdx.x;
    const int32_t gridSize = blockSize * gridDim.x;
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

template <uint32_t blockSize> __global__ void reduce2D(float *g_idata, float *g_odata, int32_t N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
    const uint32_t offset = blockIdx.y * N;
    const uint32_t gridSize = blockSize * 2 * gridDim.x;
    int32_t n = N - blockSize;
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
__global__ void reduce2DStrided(float *g_idata, float *g_odata, int32_t N, int32_t stride) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    uint32_t tid = threadIdx.x;
    int32_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
    const uint32_t offset = blockIdx.y;
    const uint32_t gridSize = blockSize * 2 * gridDim.x;
    int32_t n = N - blockSize;
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

void grid2D(dim3 *dimGrid) {
    // take a 1D grid that is too large and change to 2D
    int32_t x = dimGrid->x;
    int32_t y = 1;

    while(x > MAX_BLOCKS) {
        x >>= 2;
        y <<= 2;
    }
    dimGrid->y = y;
    dimGrid->x = dimGrid->x / y + (!(dimGrid->x % y) ? 0 : 1);
}
