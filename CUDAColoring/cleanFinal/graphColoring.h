#ifndef _GRAPHCOLORING_H_ 
#define _GRAPHCOLORING_H_ 

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <iostream>


const long GRAPHSIZE = 10240;    // number of nodes
const long NUMEDGES = 150000;    // number of edges 

const int GRIDSIZE = 2;          // number of blocks 
const int BLOCKSIZE = 256;       // number of threads in a block 

const int SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE); 

const int SUBSIZE_BOUNDARY = 256;

const int MAXDEGREE = 100;		// this is assumed to be true


#ifdef __cplusplus 
	#define CHECK_EXT extern "C" 
#else 
	#define CHECK_EXT 
#endif 


CHECK_EXT void cudaGraphColoring(int *adjacentList, int *compactAdjacencyList, int *vertexStartList, int *boundaryList, int *graphColors, int *degreeList, int *conflict, int boundarySize, int maxDegree);


#endif // _GRAPHCOLORING_H_ 

