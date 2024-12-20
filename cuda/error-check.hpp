#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaAssert(ans)                                                                                                \
    { cuda_error_handler((ans), __FILE__, __LINE__); }

inline void cuda_error_handler(cudaError_t code, const char *file, int line, bool abort = true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "CUDA error with message \"%s\" in %s on line %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

inline void cuda_error_handler(CUresult code, const char *file, int line, bool abort = true) {
    if(code != CUDA_SUCCESS) {
        const char *errorStr = NULL;
        cuGetErrorString(code, &errorStr);
        fprintf(stderr, "CUDA error with message \"%s\" in %s on line %d\n", errorStr, file, line);
        if(abort) exit(code);
    }
}

static const char *cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "UNKNOWN ERROR";
    }
}

inline void cuda_error_handler(cublasStatus_t code, const char *file, int line, bool abort = true) {
    if(code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error with message \"%s\" in %s on line %d\n", cublasGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}
