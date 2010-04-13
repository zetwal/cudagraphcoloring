#include "graphColoring.h"

// adjacencyMatrixD:adjacncy matrix
// colors:			storage for colors
// size: 		 	number of nodes
// subGraphSize: 	number of nodes per subgraph
// maxDegree:	    maximum degree of the graph
__global__ void colorGraph(int *adjacencyMatrixD, int *colors, int size, int subGraphSize, int maxDegree){ 
	int i, j, start, end;
	int numColors = 0; 

	start = (size/gridDim.x * blockIdx.x) + (subGraphSize/blockDim.x * threadIdx.x);	// y position of start of row
	end = start + subGraphSize;

	//int *degreeArray; 
	//degreeArray = new int[maxDegree+1]; 
	int degreeArray[50]; 
	

	for (i=start; i<end; i++) 
	{                
		// initialize degree array: stores colors used
		for (j=0; j<=maxDegree; j++) 
			degreeArray[j] = j+1; 
		
		
		// check the colors 
		for (j=start; j<end; j++){ 
			if (i == j) 
				continue; 
			
			// check connected 
			if (adjacencyMatrixD[i*size + j] == 1) 
				if (colors[j-start] != 0) 
					degreeArray[colors[j-start]-1] = 0;   // set connected spots to 0 
		} 
		

		for (j=0; j<=maxDegree; j++) 
			if (degreeArray[j-start] != 0){ 
				colors[i-start] = degreeArray[j-start]; 
				break; 
			} 
		
		if (colors[(i-start)] > numColors) 
			numColors=colors[(i-start)]; 
	} 

	//delete[] degreeArray;
} 



extern "C" __host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree)
{
	// partitioning
	int numSub = ceil((float)GRAPHSIZE/(float)SUBSIZE);
	memset(graphColors, 0, GRAPHSIZE*sizeof(int));  


	
	int k, maxColor = 1;
	int *adjacencyMatrixD, *colorsD;

	// Allocating memory on device
	cudaMalloc((void**)&adjacencyMatrixD, GRAPHSIZE*GRAPHSIZE * sizeof(int));
	cudaMalloc((void**)&colorsD, 		  GRAPHSIZE * sizeof(int));


	// transfer data to destination [ device(GPU) ] from source [ host(CPU) ]
	cudaMemcpy(adjacencyMatrixD, adjacencyMatrix, GRAPHSIZE*GRAPHSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(colorsD, 			 graphColors,     GRAPHSIZE * sizeof(int),        cudaMemcpyHostToDevice);


	// graph coloring
	dim3 dimGrid( ceil(GRAPHSIZE/(float)BLOCKSIZE_Y) );
	dim3 dimBlock( BLOCKSIZE_Y );
	colorGraph<<<dimGrid, dimBlock>>>(adjacencyMatrixD, colorsD, GRAPHSIZE, SUBSIZE, maxDegree);


	// transfer result to destination[ host(CPU) ] from source from [ device(GPU) ]
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE * sizeof(int), cudaMemcpyDeviceToHost);


	// free memory
	cudaFree(adjacencyMatrixD);
	cudaFree(colorsD);

	

	//cout<<"partitioned graphColors:"<<endl;	
	printf("Partitioned graph colors: \n"); 
	for (k=0; k<GRAPHSIZE; k++) 
		//cout << graphColors[k] << "  "; 
		printf("%d ", graphColors[k]);
	
	printf("\n"); 
	//cout << endl; 
	//cout<<"number of colors:"<< maxColor << endl;
}
