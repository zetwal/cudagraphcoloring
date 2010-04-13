#ifndef _GRAPHCOLORING_H_
#define _GRAPHCOLORING_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


const int GRAPHSIZE = 32; 
const int SUBSIZE = 4;

#define GRIDSIZE 2
#define BLOCKSIZE 4			// number of threads in a block

#ifdef __cplusplus
	#define CHECK_EXT extern "C"
#else
	#define CHECK_EXT
#endif


CHECK_EXT __host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree);


#endif // _GRAPHCOLORING_H_
