### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part B Assignment 4

SDK_INSTALL_PATH :=  C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L"$(SDK_INSTALL_PATH)/lib/x64" -L"$(SDK_INSTALL_PATH)/bin"
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
# EXECS :=  vecadd00 matmult00
EXECS :=  q1 q2
all: $(EXECS)

#######################################################################
clean:
	del /F /Q $(EXECS) *.o *.obj *.exp *.lib

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
# addKernel
addKernel.o: addKernel.cu
	${NVCC} -c -o $@ $< $(OPTIONS)

#######################################################################
# Question 1
q1 : q1.cu addKernel.h addKernel.o timer.o
	${NVCC} $< addKernel.o -o $@ ${LIB} timer.o $(OPTIONS)

#######################################################################
# Question 2
q2 : q2.cu addKernel.h addKernel.o timer.o
	${NVCC} $< addKernel.o -o $@ ${LIB} timer.o $(OPTIONS)

#######################################################################
# Question 3
q3 : q3.cu addKernel.h addKernel.o timer.o
	${NVCC} $< addKernel.o -o $@ ${LIB} timer.o $(OPTIONS)
