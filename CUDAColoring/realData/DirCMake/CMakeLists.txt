cmake_minimum_required(VERSION 2.6.2)
project (CUDAGraphColoring)

find_package(CUDA REQUIRED)


if( CUDA_CUDA_LIBRARY)
 message(STATUS " CUDA  found on this system")

	#this block sets up the shared library called prog1
	set (CUDA_NVCC_FLAGS --shared  --compiler-options;'-fPIC' )
 
	include_directories(${CMAKE_SOURCE_DIR}) 

#${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}/common/inc 
	cuda_add_library(CUDAGraphColoring SHARED ../graphColoring.cu)
	target_link_libraries(CUDAGraphColoring ${CUDA_LIBRARIES})

	include_directories(${CUDA_INCLUDE_DIRS}) 

	#this compiles and links the binary, which is called eg
	cuda_add_executable(cudaGraphColoring ../graphDriver)
	set (libs_to_link  ${CUDA_LIBRARIES} )
#/usr/local/cuda/lib64/libcudart.so /usr/lib64/libcuda.so /usr/local/cuda/lib64/libcublas.so

#${CUDA_SDK_ROOT_DIR}/lib/libcutil.a

	target_link_libraries(cudaGraphColoring CUDAGraphColoring ${libs_to_link})


else( CUDA_CUDA_LIBRARY)
	 message(STATUS "oops CUDA not found on this system")
endif(CUDA_CUDA_LIBRARY)
