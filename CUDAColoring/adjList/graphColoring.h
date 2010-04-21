#ifndef _GRAPHCOLORING_H_ 
#define _GRAPHCOLORING_H_ 

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <cuda_runtime_api.h> 
#include <cuda.h> 
#include <iostream>


const int GRAPHSIZE = 10240;    // number of nodes
const int NUMEDGES = 15000;    // number of edges 

#define GRIDSIZE 4                      // number of blocks 
#define BLOCKSIZE 512                  // number of threads in a block 

const int SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE); 

const int SUBSIZE_BOUNDARY = 256;

#ifdef __cplusplus 
#define CHECK_EXT extern "C" 
#else 
#define CHECK_EXT 
#endif 


CHECK_EXT __host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree); 
CHECK_EXT void colorConfilctDetection(int *adjacencyMatrix, int *boundaryList, int *graphColors, int *conflict, int boundarySize); 
CHECK_EXT void colorAndConflict(int *adjacencyMatrix, int *boundaryList, int *graphColors, int *conflict, int boundarySize, int maxDegree);
CHECK_EXT void cudaGraphColoring(int *adjacentList, int *boundaryList, int *graphColors, int *conflict, int boundarySize, int maxDegree);
#endif // _GRAPHCOLORING_H_ 

