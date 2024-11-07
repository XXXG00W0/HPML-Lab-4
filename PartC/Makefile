### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part C Assignment 4

SDK_INSTALL_PATH := C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
CUDNN_PATH := C:/Program Files/NVIDIA/CUDNN/v9.5
NVCC := "$(SDK_INSTALL_PATH)/bin/nvcc"
LIB := -L"$(SDK_INSTALL_PATH)/lib/x64" -L"$(SDK_INSTALL_PATH)/bin" -L"$(CUDNN_PATH)/lib/11.8/x64" -lcudnn
INCLUDES := -I"$(SDK_INSTALL_PATH)/include" -I"$(SDK_INSTALL_PATH)/samples/common/inc" -I"$(CUDNN_PATH)/include/11.8"
OPTIONS := -O3
#--maxrregcount=100 --ptxas-options -v

TAR_FILE_NAME := YourNameCUDA1.tar
EXECS := convolution
all: $(EXECS)

#######################################################################
clean:
	del /F /Q $(EXECS) *.o *.obj *.exp *.lib

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt

#######################################################################

convolution : convolution.cu
	$(NVCC) $< -o $@ $(LIB) $(INCLUDES) $(OPTIONS)