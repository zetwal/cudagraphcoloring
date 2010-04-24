#ifndef _GRAPHCOLORING_H_ 
#define _GRAPHCOLORING_H_ 

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <iostream>

long graphsize;

const long GRAPHSIZE = 141120;    // number of nodes
const long NUMEDGES = 705600;    // number of edges 

const int GRIDSIZE = 8;          // number of blocks 
const int BLOCKSIZE = 512;       // number of threads in a block 

const int SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE); 

const int SUBSIZE_BOUNDARY = 256;


#ifdef __cplusplus 
	#define CHECK_EXT extern "C" 
#else 
	#define CHECK_EXT 
#endif 


CHECK_EXT void cudaGraphColoring(int *adjacentList, int *boundaryList, int *graphColors, int *degreeList, int *conflict, int boundarySize, int maxDegree);


#endif // _GRAPHCOLORING_H_ 

