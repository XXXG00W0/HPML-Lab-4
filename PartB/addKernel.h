// addKernel.h

typedef struct{
    int width;
    int height;
    float* elements;
} Matrix;

__host__ void arrayAddHost(const float* A, const float* B, float* C, int length);
__global__ void arrayAddKernel(const float* A, const float* B, float* C, int threadPerBlock);