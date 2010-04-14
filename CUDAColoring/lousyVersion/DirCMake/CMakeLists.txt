cmake_minimum_required(VERSION 2.6.2)
project (CUDAGraphColoring)

find_package(CUDA REQUIRED)


if( CUDA_CUDA_LIBRARY)
 message(STATUS " CUDA  found on this system")

	#this block sets up the shared library called prog1
	set (CUDA_NVCC_FLAGS --shared  --compiler-options;'-fPIC' )
 
	include_directories(${CMAKE_SOURCE_DIR} /usr/local/cuda/include) 

#${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}/common/inc 

	include_directories(${CUDA_INCLUDE_DIRS}) 

	#this compiles and links the binary, which is called eg
	cuda_add_executable(cudaGraphColoring ../graphColoring.cu ../graphDriver)


else( CUDA_CUDA_LIBRARY)
	 message(STATUS "oops CUDA not found on this system")
endif(CUDA_CUDA_LIBRARY)
