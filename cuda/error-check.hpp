#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
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
