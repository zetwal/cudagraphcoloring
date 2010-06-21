#ifndef _GRAPHCOLORING_H_ 
#define _GRAPHCOLORING_H_ 

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <iostream>
using namespace std;



//const long GRAPHSIZE = 4096;    // number of nodes
//const float DENSITY = 0.01;
//const long NUMEDGES = DENSITY*GRAPHSIZE*(GRAPHSIZE-1)/2;



//const int GRIDSIZE = 4;				// number of blocks 
//const int BLOCKSIZE = 64;			// number of threads in a block 

//const int SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE); 

//const int SUBSIZE_BOUNDARY = 256;


#ifdef __cplusplus 
	#define CHECK_EXT extern "C" 
#else 
	#define CHECK_EXT 
#endif 


CHECK_EXT void cudaGraphColoring(int *adjacentList, int *boundaryList, int *graphColors, int *degreeList, 
								 int *conflict, int boundarySize, int maxDegree, int graphSize, 
								 int passes, int subsizeBoundary, int _gridSize, int _blockSize, 
								 int *startPartitionList, int *endPartitionList, int *randomList, int numRand);


#endif // _GRAPHCOLORING_H_ 

