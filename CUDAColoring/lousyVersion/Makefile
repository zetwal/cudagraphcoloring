# CUDA Makefile by Pascal Grosset
# OS: Linux/Max OSX for cuda 2.3
# Date: Jan 29 2010


# Compilation tools and exe
EXEC = cuExe
CC = g++
NVCC = nvcc


# Compile options
CFLAGS = -c -m32 -Wall
NVCFLAGS = -m32 --shared --compiler-options -fPIC


# Lib and include
CUDA_LIB=/usr/local/cuda/lib
CUDA_INCLUDE=/usr/local/cuda/include
CURRENT_LIB=.

FLAGS = -I$(CUDA_INCLUDE) -L$(CURRENT_LIB) -lmyCuda -L$(CUDA_LIB) -lcuda -lcudart -lm 



all: graphDriver.o libmyCuda.so
	$(CC) graphDriver.o  $(FLAGS) -o $(EXEC) -arch i386  #MacOSX
	# $(CC) graphDriver.o  $(FLAGS) -o $(EXEC)  #Linux
	


graphDriver.o: graphDriver.cpp graphColoring.h
	$(CC) $(CFLAGS) -I$(CUDA_INCLUDE) graphDriver.cpp

libmyCuda.so: graphColoring.h graphColoring.cu
	$(NVCC) $(NVCFLAGS) graphColoring.cu -o libmyCuda.so



clean:
	rm -rf *.o *.so cuExe
	clear





