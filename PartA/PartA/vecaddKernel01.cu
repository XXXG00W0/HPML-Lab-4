# include <stdio.h>

__global__ void AddVectors(const float* A, const float* B, float* C, int ValuesPerThread)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * ValuesPerThread;
    int n;
    for(n = 0; n < ValuesPerThread; ++n){
        C[i + n] = A[i + n] + B[i + n];
    }
}