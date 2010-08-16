cmake_minimum_required(VERSION 2.6.2)
project (gc)

find_package(CUDA REQUIRED)

if( CUDA_CUDA_LIBRARY)
 message(STATUS " CUDA  found on this system")

	#this block sets up the shared library called prog1
	set (CUDA_NVCC_FLAGS --shared  --compiler-options;'-fPIC' )
 
	include_directories(${CUDA_INCLUDE_DIRS} ) 
	cuda_add_library(gcKernel SHARED graphColoring.cu)
	#target_link_libraries(gcKernel ${CUDA_LIBRARIES} ${CUDA_SDK_ROOT_DIR}/lib/libcutil.a)
	target_link_libraries(gcKernel ${CUDA_LIBRARIES} )

	#this compiles and links the binary, which is called eg
	include_directories(${CMAKE_SOURCE_DIR}) 
	cuda_add_executable(gc graphDriver.cpp tree.h tree.cpp)
	set (libs_to_link gcKernel ${CUDA_LIBRARIES})
	target_link_libraries(gc ${libs_to_link})
	
	


else( CUDA_CUDA_LIBRARY)
	 message(STATUS "oops CUDA not found on this system")
endif(CUDA_CUDA_LIBRARY)
