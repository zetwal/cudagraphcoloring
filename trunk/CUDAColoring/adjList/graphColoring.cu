#include "graphColoring.h"
using namespace std;

//Author: Shusen

// returns the degree of that node
int __device__ degree(int vertex, int *degreeList){
	return degreeList[vertex];
}

int __device__ saturation(int vertex, int *adjacencyList, int *graphColors, int maxDegree){
	int saturation = 0;
	int colors[100];

	//memset(colors, 0, (maxDegree+1)*sizeof(int));		// initialize array


	for (int i=0; i<maxDegree; i++){
		if (adjacencyList[vertex*maxDegree + i] != -1)
			colors[ graphColors[vertex] ] = 1;			// at each colored set the array to 1
		else
			break;
	}


	for (int i=1; i<maxDegree+1; i++)					// count the number of 1's but skip uncolored
		if (colors[i] == 1)
			saturation++;

	return saturation;
}



// colors the vertex with the min possible color
int __device__ color(int vertex, int *adjacencyList, int *graphColors, int maxDegree, int numColored){
	int colors[100];
	
	if (graphColors[vertex] == 0)
		numColored++;
	
	for (int i=0; i<maxDegree; i++)						// set the index of the color to 1
		if (adjacencyList[vertex*maxDegree + i] != -1)
			colors[  graphColors[  adjacencyList[vertex*maxDegree + i]  ]  ] = 1;
		else {
			break;
		}

	

	for (int i=1; i<maxDegree+1; i++)					// nodes still equal to 0 are unassigned
		if (colors[i] != 1){
			graphColors[vertex] = i;
			break;
		}
	
	return numColored;
}

__global__ void colorGraphAdjL_complex(int *adjacencyList, int *graphColors, int *degreeList, int sizeGraph, int maxDegree)
{
	int i, j, start, end;
	int subGraphSize, numColored = 0;
	int satDegree, max, index;
	
	subGraphSize = sizeGraph/(gridDim.x * blockDim.x);
	start = (sizeGraph/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;

	while (numColored < subGraphSize){
		max = -1;
		
		for (int i=start; i<end; i++){
			if (graphColors[i] == 0)			// not colored
			{
				satDegree = saturation(i,adjacencyList,graphColors, maxDegree);

				if (satDegree > max){
					max = satDegree;
					index = i;				
				}

				if (satDegree == max){
					if (degree(i,degreeList) > degree(index,degreeList))
						index = i;
				}
			}

			numColored = color(index,adjacencyList,graphColors, maxDegree, numColored);
			//iterations++;
		}
	}


}


//Author: Pascal
__global__ void colorGraphAdjL(int *adjacencyListD, int *colors, int size, int maxDegree){
	int i, j, start, end;
	int subGraphSize, numColors = 0;
	
	subGraphSize = size/(gridDim.x * blockDim.x);
	start = (size/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;
	

	int degreeArray[100];
	for(i=start; i<end; i++)
	{
		for(j=0; j<maxDegree; j++)
			degreeArray[j] = j+1;


		for (j=0; j<maxDegree; j++){
			int vertexNeigh = i*maxDegree + j;

			if (adjacencyListD[vertexNeigh] != -1){
				if (colors[ adjacencyListD[vertexNeigh] ] != 0)
					degreeArray[ colors[adjacencyListD[vertexNeigh]] -1 ] = 0;
			}
			else
				break;
		}
		

		for(j=0; j<maxDegree; j++)
			if(degreeArray[j] != 0){
				colors[i] = degreeArray[j];
				break;
			}
		
		if(colors[i] > numColors)
			numColors = colors[i];		
	}
}



//Author: Pascal
__global__ void colorGraph(int *adjacencyMatrixD, int *colors, int size, int maxDegree){
	int i, j, start, end;
	int subGraphSize, numColors = 0;
	
	subGraphSize = size/(gridDim.x * blockDim.x);
	start = (size/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;
	
	int degreeArray[100];
	
	for(i=start; i<end; i++)
	{
		for(j=0; j<=maxDegree; j++)
			degreeArray[j] = j+1;
		
		for(j=start; j<end; j++){
			if(i==j)
			    continue;
			
			if(adjacencyMatrixD[i*size + j] == 1)
				if(colors[j] != 0)
					degreeArray[colors[j]-1] = 0;
		}	   
		
		for(j=0; j<=maxDegree; j++)
			if(degreeArray[j] != 0){
				colors[i] = degreeArray[j];
				break;
			}
		
		if(colors[i] > numColors)
			numColors = colors[i];		
	}
}


//Author: Pascal
extern "C"
__host__ void subGraphColoring(int *adjacencyMatrix, int *graphColors, int maxDegree)
{
	int *adjacencyMatrixD, *colorsD;
	
	cudaMalloc((void**)&adjacencyMatrixD, GRAPHSIZE*GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&colorsD, GRAPHSIZE*sizeof(int));
	
	cudaMemcpy(adjacencyMatrixD, adjacencyMatrix, GRAPHSIZE*GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colorsD, graphColors, GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 dimGrid(GRIDSIZE);
	dim3 dimBlock(BLOCKSIZE);
	
	colorGraph<<<dimGrid, dimBlock>>>(adjacencyMatrixD, colorsD, GRAPHSIZE, maxDegree);
	
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(adjacencyMatrixD);
	cudaFree(colorsD);
	
}


// Author :Peihong
__global__ void detectConflicts(int *adjacencyMatrixD, int *colors, int *conflictD, int size){
	int i, j, start, end;
	int subGraphSize, numColors = 0;
	
	subGraphSize = size/(gridDim.x * blockDim.x);
	start = (size/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;
	
	if(end > size) end = size;
	
	for(i=start; i<end; i++)
	{
		for(j=end; j < size; j++)
		{
			if(adjacencyMatrixD[i*size + j] == 1 && (colors[i] == colors[j]))
			{
				conflictD[min(i,j)] = 1;	
			}
		}
	}
	
}

//Author: Peihong
__global__ void detectConflicts(int *adjacencyMatrixD, int *boundaryListD, int *colors, int *conflictD, int size, int boundarySize){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i;
	if(idx < boundarySize){
		i = boundaryListD[idx];
		conflictD[idx] = 0;
		for(int j= i+1; j < size; j++)
		{
			if(adjacencyMatrixD[j*size + i] == 1 && (colors[i] == colors[j]))
			{
				//conflictD[idx] = min(i,j)+1;	
				conflictD[idx] = i+1;	
			}		
		}
	}
}


/*__global__ void detectConflicts(int *adjacencyMatrixD, int *boundaryListD, int *colors, int *conflictD, int size, int boundarySize){
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 
 if(i < size){
 
 for(int idx= 0; idx < boundarySize; idx++)
 {
 int j =  boundaryListD[idx];
 conflictD[idx] = 0;
 if( adjacencyMatrixD[j*size + i] == 1 && (colors[i] == colors[j]))
 {
 conflictD[idx] = j+1;	
 }		
 }
 }
 __syncthreads();
 }*/



// Author:Peihong
extern "C"
void colorConfilctDetection(int *adjacencyMatrix, int *boundaryList, int *graphColors, int *conflict, int boundarySize)
{
	int *adjacencyMatrixD, *colorsD, *conflictD, *boundaryListD;
	
	/**
	 cudaEvent_t start, stop;         
	 float elapsedTimeCPU; 
	 
	 cudaEventCreate(&start); 
     cudaEventCreate(&stop); 
     cudaEventRecord(start, 0);       
	 /**/      
	
	cudaMalloc((void**)&adjacencyMatrixD, GRAPHSIZE*GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&colorsD, GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&conflictD, boundarySize*sizeof(int));
	cudaMalloc((void**)&boundaryListD, boundarySize*sizeof(int));
	
	cudaMemcpy(adjacencyMatrixD, adjacencyMatrix, GRAPHSIZE*GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colorsD, graphColors, GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryListD, boundaryList, boundarySize*sizeof(int), cudaMemcpyHostToDevice);
	
	
	/**
	 cudaEventRecord(stop, 0); 
	 cudaEventSynchronize(stop); 
	 cudaEventElapsedTime(&elapsedTimeCPU, start, stop); 
	 
	 cout << "GPU time: " << elapsedTimeCPU << endl; 
	 /**/
	
	
    int gridsize = ceil((float)boundarySize/(float)SUBSIZE_BOUNDARY);
	int blocksize = SUBSIZE_BOUNDARY;
	
	
	dim3 dimGrid(gridsize);
	dim3 dimBlock(blocksize);
	
	detectConflicts<<<dimGrid, dimBlock>>>(adjacencyMatrixD, boundaryListD, colorsD, conflictD, GRAPHSIZE, boundarySize);
	
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(conflict, conflictD, boundarySize*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(adjacencyMatrixD);
	cudaFree(colorsD);
	cudaFree(conflictD);
	cudaFree(boundaryListD);
}




// Author:Peihong & Pascal
// Description: Merging of colorConfilctDetection & subGraphColoring
//				to save on data transfer time
extern "C"
void colorAndConflict(int *adjacencyMatrix, int *boundaryList, int *graphColors, int *conflict, int boundarySize, int maxDegree)
{
	int *adjacencyMatrixD, *colorsD, *conflictD, *boundaryListD;     
	int gridsize = ceil((float)boundarySize/(float)SUBSIZE_BOUNDARY);
	int blocksize = SUBSIZE_BOUNDARY;
	
	cudaEvent_t start_col, start_confl, stop_col, stop_confl, start_mem, stop_mem;         
    float elapsedTime_memory, elapsedTime_col, elapsedTime_confl; 
	
	
	// memory transfer
	cudaEventCreate(&start_mem); 
    cudaEventCreate(&stop_mem); 
    cudaEventRecord(start_mem, 0); 
	
	cudaMalloc((void**)&adjacencyMatrixD, GRAPHSIZE*GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&colorsD, GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&conflictD, boundarySize*sizeof(int));
	cudaMalloc((void**)&boundaryListD, boundarySize*sizeof(int));
	
	cudaMemcpy(adjacencyMatrixD, adjacencyMatrix, GRAPHSIZE*GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colorsD, graphColors, GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryListD, boundaryList, boundarySize*sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaEventRecord(stop_mem, 0); 
    cudaEventSynchronize(stop_mem); 
	
	
	
	
	dim3 dimGrid_col(GRIDSIZE);
	dim3 dimBlock_col(BLOCKSIZE);
	
	dim3 dimGrid_confl(gridsize);
	dim3 dimBlock_confl(blocksize);
	
	
	// Graph coloring
	cudaEventCreate(&start_col); 
    cudaEventCreate(&stop_col); 
    cudaEventRecord(start_col, 0); 
	
	colorGraph<<<dimGrid_col, dimBlock_col>>>(adjacencyMatrixD, colorsD, GRAPHSIZE, maxDegree);
	
	
	cudaEventRecord(stop_col, 0); 
    cudaEventSynchronize(stop_col); 

	
	// Conflict resolution
	cudaEventCreate(&start_confl); 
    cudaEventCreate(&stop_confl); 
    cudaEventRecord(start_confl, 0); 
	
	detectConflicts<<<dimGrid_confl, dimBlock_confl>>>(adjacencyMatrixD, boundaryListD, colorsD, conflictD, GRAPHSIZE, boundarySize);
	
	cudaEventRecord(stop_confl, 0); 
    cudaEventSynchronize(stop_confl); 
	
	cudaEventElapsedTime(&elapsedTime_memory, start_mem, stop_mem); 
	cudaEventElapsedTime(&elapsedTime_col, start_col, stop_col); 
	cudaEventElapsedTime(&elapsedTime_confl, start_confl, stop_confl); 
	cout << "GPU time ~ Memory: " << elapsedTime_memory  << "  Color: " << elapsedTime_col << "  Conflict: " << elapsedTime_confl << endl; 
	
	
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(conflict, conflictD, boundarySize*sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(adjacencyMatrixD);
	cudaFree(colorsD);
	cudaFree(conflictD);
	cudaFree(boundaryListD);
}






//Author: Peihong
__global__ void conflictsDetection(int *adjacentListD, int *boundaryListD, int *colors, int *conflictD, int size, int boundarySize, int maxDegree){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j;
	if(idx < boundarySize){
		i = boundaryListD[idx];
		conflictD[idx] = 0;
		for(int k= 0; k < maxDegree; k++)
		{
			j = adjacentListD[i*maxDegree + k];
			if(j>=0 && (colors[i] == colors[j]))
			{
				//conflictD[idx] = min(i,j)+1;	
				conflictD[idx] = i+1;	
			}		
		}
	}
}

// Author:Peihong & Pascal
// Description: Merging of colorConfilctDetection & subGraphColoring
//				to save on data transfer time
extern "C"
void cudaGraphColoring(int *adjacentList, int *boundaryList, int *graphColors, int *conflict, int boundarySize, int maxDegree)
{
	int *adjacentListD, *colorsD, *conflictD, *boundaryListD;     
	int gridsize = ceil((float)boundarySize/(float)SUBSIZE_BOUNDARY);
	int blocksize = SUBSIZE_BOUNDARY;
	
	cudaEvent_t start_col, start_confl, stop_col, stop_confl, start_mem, stop_mem;         
    float elapsedTime_memory, elapsedTime_col, elapsedTime_confl; 
	
/*
	// Adj list display
	for (int i=0; i<10; i++){
		for (int j=0; j<maxDegree; j++){
			cout << adjacentList[i*maxDegree + j] << " ";
		}
		cout << endl;
	}
*/

	cout << "Max deg: " << maxDegree << endl;


	// memory transfer
	cudaEventCreate(&start_mem); 
    cudaEventCreate(&stop_mem); 
    cudaEventRecord(start_mem, 0); 
	
	cudaMalloc((void**)&adjacentListD, GRAPHSIZE*maxDegree*sizeof(int));
	cudaMalloc((void**)&colorsD, GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&conflictD, boundarySize*sizeof(int));
	cudaMalloc((void**)&boundaryListD, boundarySize*sizeof(int));
	
	cudaMemcpy(adjacentListD, adjacentList, GRAPHSIZE*maxDegree*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(colorsD, graphColors, GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryListD, boundaryList, boundarySize*sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaEventRecord(stop_mem, 0); 
    cudaEventSynchronize(stop_mem); 
	
	
	dim3 dimGrid_col(GRIDSIZE);
	dim3 dimBlock_col(BLOCKSIZE);
	
	dim3 dimGrid_confl(gridsize);
	dim3 dimBlock_confl(blocksize);
	
	
	// Graph coloring
	cudaEventCreate(&start_col); 
        cudaEventCreate(&stop_col); 
        cudaEventRecord(start_col, 0); 
	
	//colorGraph<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, GRAPHSIZE, maxDegree);
	colorGraphAdjL<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, GRAPHSIZE, maxDegree);
	
	
	cudaEventRecord(stop_col, 0); 
        cudaEventSynchronize(stop_col); 

	
	// Conflict resolution
	cudaEventCreate(&start_confl); 
        cudaEventCreate(&stop_confl); 
        cudaEventRecord(start_confl, 0); 
	
	conflictsDetection<<<dimGrid_confl, dimBlock_confl>>>(adjacentListD, boundaryListD, colorsD, conflictD, GRAPHSIZE, boundarySize, maxDegree);
	
	cudaEventRecord(stop_confl, 0); 
    cudaEventSynchronize(stop_confl); 
	
	cudaEventElapsedTime(&elapsedTime_memory, start_mem, stop_mem); 
	cudaEventElapsedTime(&elapsedTime_col, start_col, stop_col); 
	cudaEventElapsedTime(&elapsedTime_confl, start_confl, stop_confl); 
	cout << "GPU time ~ Memory: " << elapsedTime_memory  << "  Color: " << elapsedTime_col << "  Conflict: " << elapsedTime_confl << endl; 
	
	
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(conflict, conflictD, boundarySize*sizeof(int), cudaMemcpyDeviceToHost);

	

	cudaFree(adjacentListD);
	cudaFree(colorsD);
	cudaFree(conflictD);
	cudaFree(boundaryListD);
}


extern "C"
void cudaGraphColoring_complex(int *adjacentList, int *boundaryList, int *graphColors, int *degreeList, int *conflict, int boundarySize, int maxDegree)
{
	int *adjacentListD, *colorsD, *conflictD, *boundaryListD, *degreeListD;     
	int gridsize = ceil((float)boundarySize/(float)SUBSIZE_BOUNDARY);
	int blocksize = SUBSIZE_BOUNDARY;
	
	cudaEvent_t start_col, start_confl, stop_col, stop_confl, start_mem, stop_mem;         
    float elapsedTime_memory, elapsedTime_col, elapsedTime_confl; 
	
/*
	// Adj list display
	for (int i=0; i<10; i++){
		for (int j=0; j<maxDegree; j++){
			cout << adjacentList[i*maxDegree + j] << " ";
		}
		cout << endl;
	}
*/

	cout << "Max deg: " << maxDegree << endl;


	// memory transfer
	cudaEventCreate(&start_mem); 
    cudaEventCreate(&stop_mem); 
    cudaEventRecord(start_mem, 0); 
	
	cudaMalloc((void**)&adjacentListD, GRAPHSIZE*maxDegree*sizeof(int));
	cudaMalloc((void**)&colorsD, GRAPHSIZE*sizeof(int));
	cudaMalloc((void**)&conflictD, boundarySize*sizeof(int));
	cudaMalloc((void**)&boundaryListD, boundarySize*sizeof(int));
	cudaMalloc((void**)&degreeListD, GRAPHSIZE*sizeof(int));
	
	cudaMemcpy(adjacentListD, adjacentList, GRAPHSIZE*maxDegree*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(colorsD, graphColors, GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryListD, boundaryList, boundarySize*sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(degreeListD, degreeList, GRAPHSIZE*sizeof(int), cudaMemcpyHostToDevice);
	
	
	cudaEventRecord(stop_mem, 0); 
    cudaEventSynchronize(stop_mem); 
	
	
	dim3 dimGrid_col(GRIDSIZE);
	dim3 dimBlock_col(BLOCKSIZE);
	
	dim3 dimGrid_confl(gridsize);
	dim3 dimBlock_confl(blocksize);
	
	
	// Graph coloring
	cudaEventCreate(&start_col); 
        cudaEventCreate(&stop_col); 
        cudaEventRecord(start_col, 0); 
	
	//colorGraph<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, GRAPHSIZE, maxDegree);
	//colorGraphAdjL<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, GRAPHSIZE, maxDegree);
	colorGraphAdjL_complex<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, degreeListD,GRAPHSIZE, maxDegree);
	
	
	cudaEventRecord(stop_col, 0); 
        cudaEventSynchronize(stop_col); 

	
	// Conflict resolution
	cudaEventCreate(&start_confl); 
        cudaEventCreate(&stop_confl); 
        cudaEventRecord(start_confl, 0); 
	
	conflictsDetection<<<dimGrid_confl, dimBlock_confl>>>(adjacentListD, boundaryListD, colorsD, conflictD, GRAPHSIZE, boundarySize, maxDegree);
	
	cudaEventRecord(stop_confl, 0); 
    cudaEventSynchronize(stop_confl); 
	
	cudaEventElapsedTime(&elapsedTime_memory, start_mem, stop_mem); 
	cudaEventElapsedTime(&elapsedTime_col, start_col, stop_col); 
	cudaEventElapsedTime(&elapsedTime_confl, start_confl, stop_confl); 
	cout << "GPU time ~ Memory: " << elapsedTime_memory  << "  Color: " << elapsedTime_col << "  Conflict: " << elapsedTime_confl << endl; 
	
	
	cudaMemcpy(graphColors, colorsD, GRAPHSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(conflict, conflictD, boundarySize*sizeof(int), cudaMemcpyDeviceToHost);

	

	cudaFree(adjacentListD);
	cudaFree(colorsD);
	cudaFree(conflictD);
	cudaFree(boundaryListD);
}










