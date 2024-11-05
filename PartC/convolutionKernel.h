// convolutionKernel.h

#ifndef CONV_H
#define CONV_H

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

__global__ void basicConvolution2d(const InputTensor in_tensor, const FilterTensor filter, OutputTensor out_tensor);

__host__ InputTensor tensorPadding(const InputTensor in_tensor);

__host__ InputTensor buildHostInputTensor(int C, int H, int W);

__host__ InputTensor buildDeviceInputTensor(InputTensor in_tensor, bool copy);

__host__ FilterTensor buildHostFilterTensor(int K, int C, int FH, int FW);

__host__ FilterTensor buildDeviceFilterTensor(FilterTensor f_tensor, bool copy);

__host__ OutputTensor buildHostOutputTensor(int K, int H, int W);

__host__ OutputTensor buildDeviceOutputTensor(OutputTensor out_tensor, bool copy);

__host__ void initInputTensor(InputTensor in_tensor);

__host__ void initFilterTensor(FilterTensor f_tensor);

__host__ double checksum(OutputTensor out_tensor);

#endif // CONV_H