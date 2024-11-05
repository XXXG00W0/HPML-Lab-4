// Part C


#include "convolutionKernel.h"
#include "timer.h"
#include <stdio.h>
#include <cuda_runtime.h>


int C = 3;
int H = 1024;
int W = 1024;
int FW = 3;
int FH = 3;
int K = 64;

InputTensor h_in_tensor;
FilterTensor h_filter_tensor;
OutputTensor h_out_tensor;
InputTensor d_in_tensor;
FilterTensor d_filter_tensor;
OutputTensor d_output_tensor;

void Cleanup(bool noError);

// argument: kernel function
void runConvolution(void (*convKernel)(InputTensor, FilterTensor, OutputTensor)){
    
    h_in_tensor = buildHostInputTensor(C, H, W);
    h_filter_tensor = buildHostFilterTensor(K, C, FH, FW);
    h_out_tensor = buildHostOutputTensor(K, H, W);
    
    initInputTensor(h_in_tensor);
    h_in_tensor = tensorPadding(h_in_tensor);
    initFilterTensor(h_filter_tensor);

    d_in_tensor = buildDeviceInputTensor(h_in_tensor, true);
    d_filter_tensor = buildDeviceFilterTensor(h_filter_tensor, true);
    d_output_tensor = buildDeviceOutputTensor(d_output_tensor, false);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h_in_tensor.H+BLOCK_SIZE-1)/BLOCK_SIZE, (h_in_tensor.W+BLOCK_SIZE-1)/BLOCK_SIZE, K);

    //warm up
    basicConvolution2d<<<dimGrid, dimBlock>>>(d_in_tensor, d_filter_tensor, d_output_tensor);

    initialize_timer();
    start_timer();
    basicConvolution2d<<<dimGrid, dimBlock>>>(d_in_tensor, d_filter_tensor, d_output_tensor);
    cudaThreadSynchronize();
    stop_timer();
    double time = elapsed_time();
    // to-do change time to miliseconds and add a checksum algorithm to verify the output
    printf("%f*1000, %f", time, checksum(h_out_tensor));
    

}

int main(int argc, char** argv){

    printf("C1 \n");
    runConvolution(basicConvolution2d);
    
    Cleanup(true);
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_filter_tensor.elements)
        cudaFree(d_filter_tensor.elements);
    if (d_in_tensor.elements)
        cudaFree(d_in_tensor.elements);
    if (d_output_tensor.elements)
        cudaFree(d_output_tensor.elements);

    // Free host memory
    if (h_filter_tensor.elements)
        free(h_filter_tensor.elements); 
    if (h_in_tensor.elements)
        free(h_in_tensor.elements);
    if (h_out_tensor.elements)
        free(h_out_tensor.elements);
    
        
    error = cudaThreadExit();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}