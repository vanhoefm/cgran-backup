#include <grgpu_utils.h>

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, Cuda error: %s: %s.n, msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}

