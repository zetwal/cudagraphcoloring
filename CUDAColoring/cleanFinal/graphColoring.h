#ifndef _GRAPHCOLORING_H_ 
#define _GRAPHCOLORING_H_ 

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <iostream>


const long GRAPHSIZE = 2048;    // number of nodes
const long NUMEDGES = 15000;    // number of edges 

const long GRIDSIZE = 2;          // number of blocks 
const long BLOCKSIZE = 128;       // number of threads in a block 

const long SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE); 

const long SUBSIZE_BOUNDARY = 256;

const long MAXDEGREE = 100;		// this is assumed to be true


#ifdef __cplusplus 
	#define CHECK_EXT extern "C" 
#else 
	#define CHECK_EXT 
#endif 


CHECK_EXT void cudaGraphColoring(long *adjacentList, long *compactAdjacencyList, long *vertexStartList, long *boundaryList, long *graphColors, long *degreeList, long *conflict, long boundarySize, long maxDegree, long graphSize, long numEdges);


#endif // _GRAPHCOLORING_H_ 

