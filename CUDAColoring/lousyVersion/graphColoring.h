#ifndef _GRAPHCOLORING_H_
#define _GRAPHCOLORING_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


const int GRAPHSIZE = 30; 
const int SUBSIZE = 10;

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 4

#ifdef __cplusplus
	#define CHECK_EXT extern "C"
#else
	#define CHECK_EXT
#endif


CHECK_EXT __host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree);


#endif // _GRAPHCOLORING_H_
