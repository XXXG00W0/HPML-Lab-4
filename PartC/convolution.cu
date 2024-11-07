// Part C
#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <windows.h>


// define marco for error checking
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cout << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

typedef struct {
    int C, H, W;
    double* elements;
} InputTensor;

typedef struct {
    int K, C, FH, FW;
    double* elements;
} FilterTensor;

typedef struct {
    int K, W, H;
    double* elements;
} OutputTensor;

int C = 3;
int H = 1024;
int W = 1024;
int FW = 3;
int FH = 3;
int K = 64;

// InputTensor h_in_tensor;
// FilterTensor h_filter_tensor;
// OutputTensor h_out_tensor;
InputTensor d_in_tensor;
FilterTensor d_filter_tensor;
OutputTensor d_out_tensor;

void Cleanup(bool noError);

// void initTensors();
// void freeTensors();

/* initialize a timer, this must be done before you can use the timer! */
void initialize_timer ( void );

/* clear the stored values of a timer */
void reset_timer ( void );

/* start the timer */
void start_timer ( void );

/* stop the timer */
void stop_timer ( void );

/* return the elapsed time in seconds, returns -1.0 on error */
double elapsed_time ( void );

__global__ void basicConvolution2d(const InputTensor in_tensor, const FilterTensor filter, OutputTensor out_tensor);

__global__ void tiledConvolution2d(const InputTensor in_tensor, const FilterTensor filter, OutputTensor out_tensor);

__host__ InputTensor tensorPadding(const InputTensor in_tensor);

__host__ InputTensor tensorPaddingDevice(const InputTensor in_tensor);

__host__ InputTensor buildHostInputTensor(int C, int H, int W);

__host__ InputTensor buildDeviceInputTensor(InputTensor in_tensor, bool copy);

__host__ InputTensor buildDeviceInputTensorManaged(int C, int H, int W, int P);

__host__ FilterTensor buildHostFilterTensor(int K, int C, int FH, int FW);

__host__ FilterTensor buildDeviceFilterTensor(FilterTensor f_tensor, bool copy);

__host__ FilterTensor buildDeviceFilterTensorManaged(int K, int C, int FH, int FW);

__host__ OutputTensor buildHostOutputTensor(int K, int H, int W);

__host__ OutputTensor buildDeviceOutputTensor(OutputTensor out_tensor, bool copy);

__host__ OutputTensor buildDeviceOutputTensorManaged(int K, int H, int W);

__host__ void initInputTensor(InputTensor in_tensor);

__host__ void initFilterTensor(FilterTensor f_tensor);

__host__ double checksum(OutputTensor out_tensor);

__host__ double checksum(InputTensor in_tensor);

__host__ double checksum(FilterTensor f_tensor);

void runC1(InputTensor d_in_tensor, FilterTensor d_filter_tensor, OutputTensor d_out_tensor) {
    double time;
    cudaError_t err;

    std::cout << "Input tensor dimension: " << d_in_tensor.C << " x " << d_in_tensor.H << " x " << d_in_tensor.W << std::endl;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_in_tensor.W + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_in_tensor.H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Block dimension: x = " << dimBlock.x << ", y = " << dimBlock.y << std::endl;
    std::cout << "Grid dimension: x = " << dimGrid.x << ", y = " << dimGrid.y << ", z = " << dimGrid.z << std::endl;

    // C1 basic convolution 2d
    // warm up
    basicConvolution2d<<<dimGrid, dimBlock>>>(d_in_tensor, d_filter_tensor, d_out_tensor);
    cudaDeviceSynchronize();
    printf("C1 warmup finished\n");

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("C1 warmup CUDA error: %s\n", cudaGetErrorString(err));
    }

    initialize_timer();
    start_timer();
    basicConvolution2d<<<dimGrid, dimBlock>>>(d_in_tensor, d_filter_tensor, d_out_tensor);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("C1 run CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    stop_timer();
    time = elapsed_time();

    // print time in milliseconds and checksum
    printf("%f, %.3f\n", checksum(d_out_tensor), time * 1000);


    // free memory
    // if(d_filter_tensor.elements)
    //     cudaFree(d_filter_tensor.elements);
    // if(d_in_tensor.elements) 
    //     cudaFree(d_in_tensor.elements);
    // if(d_out_tensor.elements)
    //     cudaFree(d_out_tensor.elements);
}

void runC2(InputTensor d_in_tensor, FilterTensor d_filter_tensor, OutputTensor d_out_tensor) {
    double time;
    cudaError_t err;

    // for (int i = 0; i < d_out_tensor.K * d_out_tensor.H * d_out_tensor.W; i++){
    //     d_out_tensor.elements[i] = 0.0;
    // }

    // C2 tiled & shared memory convolution 2d

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((d_in_tensor.W + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_in_tensor.H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    std::cout << "Block dimension: x = " << dimBlock.x << ", y = " << dimBlock.y << std::endl;
    std::cout << "Grid dimension: x = " << dimGrid.x << ", y = " << dimGrid.y << ", z = " << dimGrid.z << std::endl;

    // warm up
    tiledConvolution2d<<<dimGrid, dimBlock>>>(d_in_tensor, d_filter_tensor, d_out_tensor);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("C2 warm up CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    initialize_timer();
    start_timer();
    tiledConvolution2d<<<dimGrid, dimBlock>>>(d_in_tensor, d_filter_tensor, d_out_tensor);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("C2 run CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    stop_timer();
    time = elapsed_time();

    // print time in milliseconds and checksum
    printf("%f, %.3f\n", checksum(d_out_tensor), time * 1000);

    // free memory
    // if(d_filter_tensor.elements)
    //     cudaFree(d_filter_tensor.elements);
    // if(d_in_tensor.elements)
    //     cudaFree(d_in_tensor.elements);
    // if(d_out_tensor.elements)
    //     cudaFree(d_out_tensor.elements);
}

void runC3(InputTensor d_in_tensor, FilterTensor d_filter_tensor, OutputTensor d_out_tensor) {
    double time;
    cudaError_t err;

    // C3 cudnn convolution 2d
    // Codes cited from 
    // https://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
    // By Peter Goldsborough

    for (int i = 0; i < d_out_tensor.K * d_out_tensor.H * d_out_tensor.W; i++){
        d_out_tensor.elements[i] = 0.0;
    }

    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronization error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Describing operands
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_DOUBLE,
                                      /*batch_size=*/1,
                                      /*channels=*/d_in_tensor.C,
                                      /*image_height=*/d_in_tensor.H-2,
                                      /*image_width=*/d_in_tensor.W-2));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_DOUBLE,
                                        /*batch_size=*/1,
                                        /*channels=*/d_out_tensor.K,
                                        /*image_height=*/d_out_tensor.H,
                                        /*image_width=*/d_out_tensor.W));

    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                        /*dataType=*/CUDNN_DATA_DOUBLE,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/d_filter_tensor.K,
                                        /*in_channels=*/d_filter_tensor.C,
                                        /*kernel_height=*/d_filter_tensor.FH,
                                        /*kernel_width=*/d_filter_tensor.FW));
    
    // Describing the Convolution Kernel
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            /*pad_height=*/1,
                                            /*pad_width=*/1,
                                            /*vertical_stride=*/1,
                                            /*horizontal_stride=*/1,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            /*mode=*/CUDNN_CROSS_CORRELATION,
                                            /*computeType=*/CUDNN_DATA_DOUBLE));

    // cudnnConvolutionFwdAlgo_t convolution_algorithm;
    // cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    // int returnedAlgoCount;
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
    //                                         input_descriptor,
    //                                         filter_descriptor,
    //                                         convolution_descriptor,
    //                                         output_descriptor,
    //                                         1,
    //                                         &returnedAlgoCount,
    //                                         perfResults));
    // convolution_algorithm = perfResults[0].algo;
    cudnnConvolutionFwdAlgo_t convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    // Allocating memory
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    filter_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes));
    std::cout << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    
    // convolution
    const float alpha = 1, beta = 0;                        

    initialize_timer();
    start_timer();
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   d_in_tensor.elements,
                                   filter_descriptor,
                                   d_filter_tensor.elements,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   d_out_tensor.elements));
    cudaDeviceSynchronize();
    stop_timer();
    time = elapsed_time();

    // cudaMemcpy(h_out_tensor.elements, d_out_tensor.elements, size, cudaMemcpyDeviceToDevice);
    // print time in milliseconds and checksum
    printf("%f, %.3f\n", checksum(d_out_tensor), time * 1000);

    // free memory
    cudaFree(d_workspace);

    if (d_filter_tensor.elements)
        cudaFree(d_filter_tensor.elements);
    if (d_in_tensor.elements)
        cudaFree(d_in_tensor.elements);
    if (d_out_tensor.elements)
        cudaFree(d_out_tensor.elements);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

    InputTensor d_in_tensor = buildDeviceInputTensorManaged(C, H, W, 1);
    FilterTensor d_filter_tensor = buildDeviceFilterTensorManaged(K, C, FH, FW);
    OutputTensor d_out_tensor = buildDeviceOutputTensorManaged(K, H, W);
    
    printf("in tensor: %f\n", checksum(d_in_tensor));
    printf("filter tensor: %f\n", checksum(d_filter_tensor));
    
    runC3(d_in_tensor, d_filter_tensor, d_out_tensor);
    runC1(d_in_tensor, d_filter_tensor, d_out_tensor);
    runC2(d_in_tensor, d_filter_tensor, d_out_tensor);
    Cleanup(true);
    return 0;
}

void Cleanup(bool noError) {  // simplified version from CUDA SDK
    cudaError_t error;
        
    // Free device vectors
    if (d_filter_tensor.elements)
        cudaFree(d_filter_tensor.elements);
    if (d_in_tensor.elements)
        cudaFree(d_in_tensor.elements);
    if (d_out_tensor.elements)
        cudaFree(d_out_tensor.elements);

    // Free host memory
    // if (h_filter_tensor.elements)
    //     free(h_filter_tensor.elements); 
    // if (h_in_tensor.elements)
    //     free(h_in_tensor.elements);
    // if (h_out_tensor.elements)
    //     free(h_out_tensor.elements);
    
        
    error = cudaThreadExit();
    
    if (!noError || error != cudaSuccess)
        printf("cuda malloc or cuda thread exit failed \n");
    
    fflush( stdout);
    fflush( stderr);

    exit(0);
}

__host__ InputTensor buildHostInputTensor(int C, int H, int W){

    InputTensor in_tensor;
    in_tensor.C = C;
    in_tensor.H = H;
    in_tensor.W = W;
    size_t size = C * H * W * sizeof(double);
    in_tensor.elements = (double*)malloc(size);
    if (in_tensor.elements == NULL)
        Cleanup(false);
    return in_tensor;
}

__host__ InputTensor buildDeviceInputTensor(InputTensor in_tensor, bool copy){
    
    cudaError_t error;

    InputTensor in_tensor_d;
    in_tensor_d.C = in_tensor.C;
    in_tensor_d.H = in_tensor.H;
    in_tensor_d.W = in_tensor.W;
    size_t size = in_tensor.C * in_tensor.H * in_tensor.W * sizeof(double);
    error = cudaMalloc((void**)&in_tensor_d.elements, size);
    if (error != cudaSuccess) Cleanup(false);
    if(copy){
        error = cudaMemcpy(in_tensor_d.elements, in_tensor.elements, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) Cleanup(false);
    }
    return in_tensor_d;
}

__host__ InputTensor buildDeviceInputTensorManaged(int C, int H, int W, int P){
    InputTensor in_tensor;
    in_tensor.C = C;
    in_tensor.H = H + 2*P;
    in_tensor.W = W + 2*P;
    size_t size = C * in_tensor.H * in_tensor.W * sizeof(double);
    cudaError_t error = cudaMallocManaged((void**)&in_tensor.elements, size);
    if (error != cudaSuccess) Cleanup(false);
    // initInputTensor(in_tensor);

    // pad input tensor
    for (int c = 0; c < in_tensor.C; c++){
        for (int h = 0; h < in_tensor.H; h++){
            for (int w = 0; w < in_tensor.W; w++){
                if (h == 0 || h == in_tensor.H - 1 || w == 0 || w == in_tensor.W - 1)
                    in_tensor.elements[c * in_tensor.H * in_tensor.W + h * in_tensor.W + w] = 0;
                else
                    in_tensor.elements[c * in_tensor.H * in_tensor.W + h * in_tensor.W + w] = c * (h-P + w-P);
            }
        }
    }
    return in_tensor;
}

__host__ FilterTensor buildHostFilterTensor(int K, int C, int FH, int FW){

    FilterTensor f_tensor;
    f_tensor.K = K;
    f_tensor.C = C;
    f_tensor.FH = FH;
    f_tensor.FW = FW;
    size_t size = K * C * FH * FW * sizeof(double);
    f_tensor.elements = (double*) malloc(size);
    if (f_tensor.elements == NULL)
        Cleanup(false);
    return f_tensor;
}

__host__ FilterTensor buildDeviceFilterTensor(FilterTensor f_tensor, bool copy){
    
    cudaError_t error;

    FilterTensor f_tensor_d;
    f_tensor_d.K = f_tensor.K;
    f_tensor_d.C = f_tensor.C;
    f_tensor_d.FH = f_tensor.FH;
    f_tensor_d.FW = f_tensor.FW;
    size_t size = f_tensor.K * f_tensor.C * f_tensor.FH * f_tensor.FW * sizeof(double);
    error = cudaMalloc((void**)&f_tensor_d.elements, size);
    if (error != cudaSuccess) Cleanup(false);
    if(copy){
        cudaError_t error = cudaMemcpy(f_tensor_d.elements, f_tensor.elements, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) Cleanup(false);
    }
    return f_tensor_d;
}

__host__ FilterTensor buildDeviceFilterTensorManaged(int K, int C, int FH, int FW){
    FilterTensor f_tensor;
    f_tensor.K = K;
    f_tensor.C = C;
    f_tensor.FH = FH;
    f_tensor.FW = FW;
    size_t size = K * C * FH * FW * sizeof(double);
    cudaError_t error = cudaMallocManaged((void**)&f_tensor.elements, size);
    if (error != cudaSuccess) Cleanup(false);
    
    // initialize filter tensor
    for (int k = 0; k < f_tensor.K; k++){
        for (int c = 0; c < f_tensor.C; c++){
            for (int fw = 0; fw < f_tensor.FW; fw++){
                for (int fh = 0; fh < f_tensor.FH; fh++){
                    f_tensor.elements[k * f_tensor.C * f_tensor.FW * f_tensor.FH + c * 
                    f_tensor.FW * f_tensor.FH + fw * f_tensor.FH + fh] = (c + k) * (fw + fh);
                }
            }
        }
    }

    return f_tensor;
}

__host__ OutputTensor buildHostOutputTensor(int K, int H, int W){

    OutputTensor out_tensor;
    out_tensor.K = K;
    out_tensor.H = H;
    out_tensor.W = W;
    size_t size = K * H * W * sizeof(double);
    out_tensor.elements = (double*)malloc(size);
    if (out_tensor.elements == NULL)
        Cleanup(false);
    return out_tensor;
}

__host__ OutputTensor buildDeviceOutputTensor(OutputTensor out_tensor, bool copy){
    
    cudaError_t error;

    OutputTensor out_tensor_d;
    out_tensor_d.K = out_tensor.K;
    out_tensor_d.H = out_tensor.H;
    out_tensor_d.W = out_tensor.W;
    size_t size = out_tensor.K * out_tensor.H * out_tensor.W * sizeof(double);
    error = cudaMalloc((void**)&out_tensor_d.elements, size);
    if (error != cudaSuccess) Cleanup(false);
    if(copy){
        error = cudaMemcpy(out_tensor_d.elements, out_tensor.elements, size, cudaMemcpyHostToDevice);
        if (error != cudaSuccess) Cleanup(false);
    }
    return out_tensor_d;
}

__host__ OutputTensor buildDeviceOutputTensorManaged(int K, int H, int W){
    OutputTensor out_tensor;
    out_tensor.K = K;
    out_tensor.H = H;
    out_tensor.W = W;
    size_t size = K * H * W * sizeof(double);
    cudaError_t error = cudaMallocManaged((void**)&out_tensor.elements, size);
    if (error != cudaSuccess) Cleanup(false);
    for (int i = 0; i < K * H * W; i++){
        out_tensor.elements[i] = 0.0;
    }
    return out_tensor;
}

__host__ void initInputTensor(InputTensor in_tensor){
    // size_t size = in_tensor.C * in_tensor.H * in_tensor.W * sizeof(double);
    double value;

    for (int c = 0; c < in_tensor.C; c++){
        for (int h = 0; h < in_tensor.H; h++){
            for (int w = 0; w < in_tensor.W; w++){
                value =  c * (h + w);
                in_tensor.elements[c * in_tensor.H * in_tensor.W + h * in_tensor.W + w] = value;
            }
        }
    }
}

__host__ void initFilterTensor(FilterTensor f_tensor){
    // size_t size = f_tensor.K * f_tensor.C * f_tensor.FW * f_tensor.FH;
    double value;

    for (int k = 0; k < f_tensor.K; k++){
        for (int c = 0; c < f_tensor.C; c++){
            for (int fw = 0; fw < f_tensor.FW; fw++){
                for (int fh = 0; fh < f_tensor.FH; fh++){
                    value = (c + k) * (fw + fh);
                    f_tensor.elements[k * f_tensor.C * f_tensor.FW * f_tensor.FH + c * 
                    f_tensor.FW * f_tensor.FH + fw * f_tensor.FH + fh] = value;
                }
            }
        }
    }
}

__host__ double checksum(OutputTensor out_tensor){
    double sum = 0.0;
    for (int i = 0; i < out_tensor.K * out_tensor.H * out_tensor.W; i++){
        sum += out_tensor.elements[i];
    }
    return sum;
}

__host__ double checksum(InputTensor in_tensor){
    double sum = 0.0;
    for (int i = 0; i < in_tensor.C * in_tensor.H * in_tensor.W; i++){
        sum += in_tensor.elements[i];
    }
    return sum;
}

__host__ double checksum(FilterTensor f_tensor){
    double sum = 0.0;
    for (int i = 0; i < f_tensor.K * f_tensor.C * f_tensor.FW * f_tensor.FH; i++){
        sum += f_tensor.elements[i];
    }
    return sum;
}

__global__ void basicConvolution2d(const InputTensor in_tensor, const FilterTensor filter, OutputTensor out_tensor){
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_tensor.W || out_y >= out_tensor.H) return;

    for (int k = 0; k < filter.K; ++k){

        double conv_sum = 0.0;
        for (int c = 0; c < in_tensor.C; ++c){
            for (int j = 0; j < filter.FH; ++j){
                for (int i = 0; i < filter.FW; ++i){

                    // calcualte the position of the filter
                    int filter_pos = k * filter.C * filter.FW * filter.FH 
                                + c * filter.FW * filter.FH
                                + (filter.FH - 1 - j) * filter.FW + (filter.FW - 1 - i);

                    // calculate the position of the input tensor
                    int in_x = out_x + i;
                    int in_y = out_y + j;
                    
                    // check if the position is valid
                    if (in_x >= 0 && in_x < in_tensor.W && in_y > 0 && in_y < in_tensor.H){
                        int in_pos = c * in_tensor.H * in_tensor.W + in_y * in_tensor.W + in_x;
                        conv_sum += filter.elements[filter_pos] * in_tensor.elements[in_pos];
                    } 
                }
            }
        } 
        int out_pos = k * out_tensor.H * out_tensor.W + out_y * out_tensor.W + out_x;
        out_tensor.elements[out_pos] = conv_sum;
    }
    
    
}

__global__ void tiledConvolution2d(const InputTensor in_tensor, const FilterTensor filter, OutputTensor out_tensor){

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double in_shared[BLOCK_SIZE+2][BLOCK_SIZE+2][3];
    // hard coded shared memory size FW * FH * C
    __shared__ double filter_shared[3][3][3][64];

    // load input tensor to shared memory
    // assumed that in_tensor is padded
    for (int c = 0; c < in_tensor.C; c++){
        int shared_x = threadIdx.x + filter.FW / 2;
        int shared_y = threadIdx.y + filter.FH / 2;
        if (out_x + shared_x >= 0 && out_x + shared_x < out_tensor.W && out_y + shared_y >= 0 && out_y + shared_y < out_tensor.H){
            in_shared[shared_x][shared_y][c] = in_tensor.elements[c * in_tensor.H * in_tensor.W + out_y * in_tensor.W + out_x];
        }else{
            in_shared[shared_x][shared_y][c] = 0.0;
        }
    }

    // load filter tensor to shared memory
    if (threadIdx.x < filter.FW && threadIdx.y < filter.FH){
        for (int c = 0; c < filter.C; c++){
            for (int k = 0; k < filter.K; k++){
                filter_shared[threadIdx.x][threadIdx.y][c][k] = filter.elements[k * filter.C * filter.FW * filter.FH
                                                                    + c * filter.FW * filter.FH 
                                                                    + threadIdx.y * filter.FW + threadIdx.x];
            }
        }
    }

    __syncthreads();

    // calculate the convolution
    if (out_x < out_tensor.W && out_y < out_tensor.H){
        for (int k = 0; k < filter.K; ++k){
            double conv_sum = 0.0;
            for (int c = 0; c < in_tensor.C; ++c){
                for (int j = 0; j < filter.FH; ++j){
                    for (int i = 0; i < filter.FW; ++i){
                        conv_sum += filter_shared[i][j][c][k] * in_shared[threadIdx.x + i][threadIdx.y + j][c];
                    }
                }
            }
            // store the result
            int out_pos = k * out_tensor.H * out_tensor.W + out_y * out_tensor.W + out_x;
            out_tensor.elements[out_pos] = conv_sum;
        }
    }

}

// __global__

// timer related functions

static double start, stop;        /* store the times locally */
static int start_flag, stop_flag; /* flag timer use */

// implement gettimeofday for windows

double gettimeofday(timeval *tp, void *tzp)
{
    LARGE_INTEGER ticksPerSecond;
    LARGE_INTEGER tick;
    QueryPerformanceFrequency(&ticksPerSecond);
    QueryPerformanceCounter(&tick);
    tp->tv_sec = tick.QuadPart / ticksPerSecond.QuadPart;
    tp->tv_usec = (tick.QuadPart % ticksPerSecond.QuadPart) * 1000000 / ticksPerSecond.QuadPart;
    return 0;
}


void initialize_timer ( void )
{
    start = 0.0;
    stop  = 0.0;

    start_flag = 0;
    stop_flag  = 0;
}


void reset_timer ( void )
{
    initialize_timer();
}


void start_timer ( void )
{
    struct timeval time;

    if ( start_flag )
	fprintf( stderr, "WARNING: timer already started!\n" );

    start_flag = 1;

    if ( gettimeofday( &time, NULL ) < 0 )
	perror( "start_timer,gettimeofday" );

    start = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
}


void stop_timer ( void )
{
    struct timeval time;

    if ( !start_flag )
	fprintf( stderr, "WARNING: timer not started!\n" );

    if ( stop_flag )
	fprintf( stderr, "WARNING: timer already stopped!\n" );

    stop_flag = 1;

    if ( gettimeofday( &time, NULL ) < 0 )
	perror( "stop_timer,gettimeofday" );

    stop = (((double) time.tv_sec) + ((double) time.tv_usec)/1000000);
}


double elapsed_time ( void )
{
    if ( !start_flag || !stop_flag )
	return (-1.0);

    return (stop-start);
}
