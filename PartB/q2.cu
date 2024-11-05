// Question 2 cudaMalloc

#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "addKernel.h"

void Cleanup(bool);
void checkCUDAError(const char *msg);

// Variables for host and device vectors.
float* h_A; 
float* h_B; 
float* h_C; 
float* d_A; 
float* d_B; 
float* d_C;

cudaError_t error;
size_t size;

void arrayAdd(int grid_size, int block_size, int arrLength, int valuesPerThread){
    
    // allocate d_C on each call
    error = cudaMalloc((void**)&d_C, size);
    if (error != cudaSuccess) Cleanup(false);
    
    double time;
    dim3 dimGrid(grid_size);
    dim3 dimBlock(block_size);

    // Warm up
    arrayAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, valuesPerThread);
    error = cudaGetLastError();
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();

    // Measure time
    initialize_timer();
    start_timer();
    arrayAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, valuesPerThread);
    if (error != cudaSuccess) Cleanup(false);
    cudaThreadSynchronize();
    stop_timer();
    time = elapsed_time();
    printf("Time measured: %g s\n", time);
    
    // Copy result from device memory to host memory
    error = cudaMemcpy(h_C, d_C, arrLength * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("%s\n", cudaGetErrorString(error));
    if (error != cudaSuccess) Cleanup(false);
    
    //Verify the result
    long i;
    for (i = 0; i < arrLength; ++i) {
        float val = h_C[i];
        if (fabs(val - arrLength) > 1e-5){
            printf("i=%d; val=%f\n ", i, val);
            break;
        }
    }
    printf("Test %s \n", (i == arrLength) ? "PASSED" : "FAILED");

    // free d_C after each run
    if (d_C) cudaFree(d_C);

}

int main(int argc, char** argv){
    int arrLength;
    int K;

    if (argc != 2){
        printf("Usage: %s <K> \n", argv[0]);
        exit(0);
    } else {
        sscanf(argv[1], "%d", &K);
    };

    arrLength = K * 1000000;
    printf("Question 2: \nArray length: %d\n", arrLength);
    size = arrLength * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup(false);
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup(false);
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup(false);

    // Allocate vectors in device memory.
    error = cudaMalloc((void**)&d_A, size);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMalloc((void**)&d_B, size);
    if (error != cudaSuccess) Cleanup(false);
    // error = cudaMalloc((void**)&d_C, size);
    // if (error != cudaSuccess) Cleanup(false);

    // Initialize host vectors h_A and h_B
    int i;
    for(i=0; i<arrLength; ++i){
        h_A[i] = (float)i;
        h_B[i] = (float)(arrLength-i);
    }

    // Copy host vectors h_A and h_B to device vectores d_A and d_B
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) Cleanup(false);

    // Using one block with one thread
    printf("One block with one thread: \n");
    arrayAdd(1, 1, arrLength, arrLength);

    // Using one block with 256 threads
    printf("One block with 256 threads: \n");
    arrayAdd(1, 256, arrLength, (arrLength + 255) / 256);

    // Using multiple blocks with 256 threads per block with the total number of threads
    // across all the blocks equal to the size of arrays
    printf("Multiple blocks with 256 threads per block: \n");
    arrayAdd((arrLength + 255) / 256, 256, arrLength, 1);

    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    error = cudaThreadExit();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
      exit(-1);
    }                         
}
