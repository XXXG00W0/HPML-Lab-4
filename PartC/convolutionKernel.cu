// conv.cu

#include "convolutionKernel.h"

__global__ void basicConvolution2d(const InputTensor in_tensor, const FilterTensor filter, OutputTensor out_tensor){
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_k = blockIdx.z;

    double conv_sum = 0.0;

    int out_pos, filter_pos, in_pos;
    for (int c = 0; c < in_tensor.C; ++c){
        for (int j = 0; j < in_tensor.H; ++j){
            for (int i = 0; i < in_tensor.W; ++i){
                filter_pos = out_k * filter.C * filter.FW * filter.FH + c * filter.FW * filter.FH + 
                            (filter.FW - 1 - i) * filter.FH + (filter.FH - 1 - j);
                in_pos = c * in_tensor.H * in_tensor.W + (out_x + i) * in_tensor.W + (out_y + j);
                conv_sum += filter.elements[filter_pos] * in_tensor.elements[in_pos];
            }
        }
    }
    
    out_pos = out_k * out_tensor.H * out_tensor.W + out_x * out_tensor.W + out_y;
    out_tensor.elements[out_pos] = conv_sum;
    
}

__global__ void tiledConvolution2d(const InputTensor in_tensor, const FilterTensor filter_tensor, OutputTensor out_tensor){
    float *in_sub, *out_sub;

    __shared__ double in_shared[BLOCK_SIZE+2][BLOCK_SIZE+2];
    // __shared__ double filter_shared

    in_shared[out_x]

    double conv_sum = 0.0;
}

__host__ InputTensor tensorPadding(const InputTensor in_tensor){
    InputTensor padded;
    padded.C = in_tensor.C;
    padded.H = in_tensor.H + 2;
    padded.W = in_tensor.W + 2;
    size_t size = padded.C * padded.H * padded.W * sizeof(double);
    padded.elements = (double*)malloc(size);
    for (int c = 0; c < padded.C; c++){
        for (int h = 0; h < padded.H; h++){
            for (int w = 0; w < padded.W; w++){
                if (h == 0 || h == padded.H - 1 || w == 0 || w == padded.W - 1){
                    padded.elements[c * padded.H * padded.W + h * padded.W + w] = 0;
                } else {
                    padded.elements[c * padded.H * padded.W + h * padded.W + w] = in_tensor.elements[c * in_tensor.H * in_tensor.W + (h - 1) * in_tensor.W + (w - 1)];
                }
            }
        }
    }
    return padded;
}

__host__ InputTensor buildHostInputTensor(int C, int H, int W){

    InputTensor in_tensor;
    in_tensor.C = C;
    in_tensor.H = H;
    in_tensor.W = W;
    size_t size = C * H * W * sizeof(double);
    in_tensor.elements = (double*)malloc(size);
    return in_tensor;
}

__host__ InputTensor buildDeviceInputTensor(InputTensor in_tensor, bool copy){
    
    InputTensor in_tensor_d;
    in_tensor_d.C = in_tensor.C;
    in_tensor_d.H = in_tensor.H;
    in_tensor_d.W = in_tensor.W;
    size_t size = in_tensor.C * in_tensor.H * in_tensor.W * sizeof(double);
    cudaMalloc((void**)&in_tensor_d.elements, size);
    if(copy){
        cudaMemcpy(in_tensor_d.elements, in_tensor.elements, size, cudaMemcpyHostToDevice);
    }
    return in_tensor_d;
}

__host__ FilterTensor buildHostFilterTensor(int K, int C, int FH, int FW){

    FilterTensor f_tensor;
    f_tensor.K = K;
    f_tensor.C = C;
    f_tensor.FH = FH;
    f_tensor.FW = FW;
    size_t size = K * C * FH * FW * sizeof(double);
    f_tensor.elements = (double*) malloc(size);
    return f_tensor;
}

__host__ FilterTensor buildDeviceFilterTensor(FilterTensor f_tensor, bool copy){
    
    FilterTensor f_tensor_d;
    f_tensor_d.K = f_tensor.K;
    f_tensor_d.C = f_tensor.C;
    f_tensor_d.FH = f_tensor.FH;
    f_tensor_d.FW = f_tensor.FW;
    size_t size = f_tensor.K * f_tensor.C * f_tensor.FH * f_tensor.FW * sizeof(double);
    cudaMalloc((void**)&f_tensor_d.elements, size);
    if(copy){
        cudaMemcpy(f_tensor_d.elements, f_tensor.elements, size, cudaMemcpyHostToDevice);
    }
    return f_tensor_d;
}

__host__ OutputTensor buildHostOutputTensor(int K, int H, int W){

    OutputTensor out_tensor;
    out_tensor.K = K;
    out_tensor.H = H;
    out_tensor.W = W;
    size_t size = K * H * W * sizeof(double);
    out_tensor.elements = (double*)malloc(size);
    return out_tensor;
}

__host__ OutputTensor buildDeviceOutputTensor(OutputTensor out_tensor, bool copy){
    
    OutputTensor out_tensor_d;
    out_tensor_d.K = out_tensor.K;
    out_tensor_d.H = out_tensor.H;
    out_tensor_d.W = out_tensor.W;
    size_t size = out_tensor.K * out_tensor.H * out_tensor.W * sizeof(double);
    cudaMalloc((void**)&out_tensor_d.elements, size);
    if(copy){
        cudaMemcpy(out_tensor_d.elements, out_tensor.elements, size, cudaMemcpyHostToDevice);
    }
    return out_tensor_d;
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
    for (int k = 0; k < out_tensor.K; k++){
        for (int h = 0; h < out_tensor.H; h++){
            for (int w = 0; w < out_tensor.W; w++){
                sum += out_tensor.elements[k * out_tensor.H * out_tensor.W + h * out_tensor.W + w];
            }
        }
    }
    return sum;
}