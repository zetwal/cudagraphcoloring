#include "graphColoring.h"

// Author: Pascal
// Break a graph into parts and color them
	// adjacencyMatrixD:adjacncy matrix
	// colors:			storage for colors
	// size: 		 	number of nodes
	// maxDegree:	    maximum degree of the graph
__global__ void colorGraph(int *adjacencyMatrixD, int *colors, int size, int maxDegree){ 
	int i, j, start, end;
	int subGraphSize, numColors = 0; 


	subGraphSize = size/(gridDim.x * blockDim.x);								// number of nodes that a thread processes
	start = (size/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);	// node starting with
	end = start + subGraphSize;

	//printf("Block: %d   Thread: %d  - start: %d   end: %d \n",blockIdx.x, threadIdx.x, start, end);
	//int *degreeArray; 
	//degreeArray = new int[maxDegree+1]; 
	int degreeArray[100]; 					// needs to change!!!
	

	for (i=start; i<end; i++) 
	{                
		// initialize degree array: stores colors that might be used
		for (j=0; j<=maxDegree; j++) 
			degreeArray[j] = j+1; 
		
		
		// check the colors 
		for (j=start; j<end; j++){ 
			if (i == j) 
				continue; 
			
			// check connected 
			if (adjacencyMatrixD[i*size + j] == 1) 
				if (colors[j] != 0) 
					degreeArray[colors[j]-1] = 0;   // set connected spots to 0 
		} 
		

		for (j=0; j<=maxDegree; j++) 
			if (degreeArray[j] != 0){ 
				colors[i] = degreeArray[j]; 
				break; 
			} 
		
		if (colors[i] > numColors) 
			numColors=colors[i]; 
	} 

	//delete[] degreeArray;
} 



// Author: Pascal
// Calls the cuda code
extern "C" __host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree)
{
	// partitioning
	int *adjacencyMatrixD, *colorsD;

//	cudaEvent_t start, stop;	
//	float elapsedTime;
	
//	cudaEventCreate(&start);//	cudaEventCreate(&stop);
//	cudaEventRecord(start, 0);
	

	// Allocating memory on device
	cudaMalloc((void**)&adjacencyMatrixD, GRAPHSIZE*GRAPHSIZE * sizeof(int));
	cudaMalloc((void**)&colorsD, 		  GRAPHSIZE * sizeof(int));


	// transfer data to destination [ device(GPU) ] from source [ host(CPU) ]
	cudaMemcpy(adjacencyMatrixD, adjacencyMatrix, GRAPHSIZE*GRAPHSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(colorsD, 			 graphColors,     GRAPHSIZE * sizeof(int),        	 cudaMemcpyHostToDevice);


	// graph coloring
	dim3 dimGrid( GRIDSIZE );
	dim3 dimBlock( BLOCKSIZE );

	

	colorGraph<<<dimGrid, dimBlock>>>(adjacencyMatrixD, colorsD, GRAPHSIZE, maxDegree);

	

	//printf("Partitioning and coloring on GPU - Time taken: %f ms\n",elapsedTime);


	// transfer result to destination[ host(CPU) ] from source from [ device(GPU) ]
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE * sizeof(int), cudaMemcpyDeviceToHost);


	// free memory
	cudaFree(adjacencyMatrixD);
	cudaFree(colorsD);

	
//	cudaEventRecord(stop, 0);//	cudaEventSynchronize(stop);//	cudaEventElapsedTime(&elapsedTime, start, stop);



// Display
/**
	printf("Partitioned graph colors: \n"); 
	for (int k=0; k<GRAPHSIZE; k++) 
		printf("%d  ", graphColors[k]);
	
	printf("\n");
/**/ 

}
