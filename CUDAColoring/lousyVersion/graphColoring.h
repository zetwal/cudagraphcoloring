#ifndef _GRAPHCOLORING_H_
#define _GRAPHCOLORING_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


const int GRAPHSIZE = 256; 	// number of nodes
const int NUMEDGES = 5000;	// number of edges

#define GRIDSIZE 2			// number of blocks
#define BLOCKSIZE 16			// number of threads in a block

const int SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE);

#ifdef __cplusplus
	#define CHECK_EXT extern "C"
#else
	#define CHECK_EXT
#endif


CHECK_EXT __host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree);


#endif // _GRAPHCOLORING_H_
