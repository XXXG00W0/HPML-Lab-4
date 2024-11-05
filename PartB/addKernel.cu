//addKernel.cu

__host__ void arrayAddHost(const float* A, const float* B, float* C, int length) {
    for (int i = 0; i < length; ++i){
        C[i] = A[i] + B[i];
    }
}

__global__ void arrayAddKernel(const float* A, const float* B, float* C, int ValuesPerThread) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * ValuesPerThread;
    int n;
    for(n = 0; n < ValuesPerThread; ++n){
        C[i + n] = A[i + n] + B[i + n];
    }
}
