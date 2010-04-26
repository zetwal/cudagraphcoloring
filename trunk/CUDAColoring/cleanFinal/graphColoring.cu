#include "graphColoring.h"
using namespace std;


//----------------------- SDO improved -----------------------//
//
// Author: Shusen & Pascal
// returns the degree of that node
int __device__ degree(long vertex, long *degreeList){
	return degreeList[vertex];
}



// Author: Shusen & Pascal
// saturation of a vertex
int __device__ saturation(long vertex, long *adjacencyList, long *graphColors, long maxDegree, long start, long end){
	int saturation = 0;	
	int colors[100];
	for (int j=0; j<100; j++)
		colors[j] = 0;


	for (int i=0; i<maxDegree; i++){
		if (adjacencyList[vertex*maxDegree + i] < start)
			continue;

		if (adjacencyList[vertex*maxDegree + i] > end)
			break;

		if (adjacencyList[vertex*maxDegree + i] != -1)
			colors[ graphColors[i] ] = 1;			// at each colored set the array to 1
		else
			break;
	}


	for (int i=1; i<maxDegree+1; i++)					// count the number of 1's but skip uncolored
		if (colors[i] == 1)
			saturation++;

	return saturation;
}




// Author: Shusen & Pascal
// colors the vertex with the min possible color
int __device__ color(long vertex, long *adjacencyList, long *graphColors, long maxDegree, long numColored, long start, long end){
	int colors[100];
	for (int j=0; j<100; j++)
		colors[j] = 0;

	
	if (graphColors[vertex] == 0)
		numColored++;
	
	for (int i=0; i<maxDegree; i++){						// set the index of the color to 1

		// Limits color checking to subgraph
/*
		if (adjacencyList[vertex*maxDegree + i] < start)
			continue;

		if (adjacencyList[vertex*maxDegree + i] > end)
			break;
*/

		if (adjacencyList[vertex*maxDegree + i] != -1)
			colors[  graphColors[  adjacencyList[vertex*maxDegree + i]  ]  ] = 1;
		else 
			break;
	}

	
	for (int i=1; i<maxDegree+1; i++)					// nodes still equal to 0 are unassigned
		if (colors[i] != 1){
			graphColors[vertex] = i;
			break;
		}
	
	return numColored;
}





// Author: Shusen & Pascal
// does the coloring
__global__ void colorGraph_SDO(long *adjacencyList, long *graphColors, long *degreeList, long sizeGraph, long maxDegree)
{
	long start, end;
	long subGraphSize, numColored = 0;
	long satDegree, max, index;
	
	subGraphSize = sizeGraph/(gridDim.x * blockDim.x);
	start = (sizeGraph/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;

	while (numColored < subGraphSize){
		max = -1;
		
		for (long i=start; i<end; i++){
			if (graphColors[i] == 0)			// not colored
			{
				satDegree = saturation(i,adjacencyList,graphColors, maxDegree, start, end);

				if (satDegree > max){
					max = satDegree;
					index = i;				
				}

				if (satDegree == max){
					if (degree(i,degreeList) > degree(index,degreeList))
						index = i;
				}
			}

			numColored = color(index,adjacencyList,graphColors, maxDegree, numColored, start, end);
		}
	}
}


//------------------------------------------------------




// Author: Shusen & Pascal
// colors the vertex with the min possible color
int __device__ color(long vertex, long *compactAdjacencyListD, long *vertexStartListD, long *graphColors, long maxDegree, long numColored, long start, long end){
	int colors[MAXDEGREE];
	for (int j=0; j<MAXDEGREE; j++)
		colors[j] = 0;

	
	if (graphColors[vertex] == 0)
		numColored++;
	

	for (long i=vertexStartListD[vertex]; i<vertexStartListD[vertex+1]; i++){
		colors[ graphColors[ compactAdjacencyListD[i] ] ] = 1;
	}

	
	for (int i=1; i<maxDegree+1; i++)					// nodes still equal to 0 are unassigned
		if (colors[i] != 1){
			graphColors[vertex] = i;
			break;
		}
	
	return numColored;
}


// Author: Shusen & Pascal
// saturation of a vertex
int __device__ saturation(long vertex, long *compactAdjacencyList, long *vertexStartList, long *graphColors, long maxDegree, long start, long end){
	int saturation = 0;	
	int colors[100];
	for (int j=0; j<100; j++)
		colors[j] = 0;



	for (long i=vertexStartList[vertex]; i<vertexStartList[vertex+1]; i++)
		colors[ graphColors[ compactAdjacencyList[i] ] ] = 1;



	for (int i=1; i<maxDegree+1; i++)					// count the number of 1's but skip uncolored
		if (colors[i] == 1)
			saturation++;

	return saturation;
}


// Author: Shusen & Pascal
// does the coloring
__global__ void colorGraph_SDO(long *compactAdjacencyList, long *vertexStartList, long *graphColors, long *degreeList, long sizeGraph, long maxDegree)
{
	long start, end;
	int subGraphSize, numColored = 0;
	int satDegree, max, index;
	
	subGraphSize = sizeGraph/(gridDim.x * blockDim.x);
	start = (sizeGraph/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;

	while (numColored < subGraphSize){
		max = -1;
		
		for (long i=start; i<end; i++){
			if (graphColors[i] == 0)			// not colored
			{
				satDegree = saturation(i, compactAdjacencyList, vertexStartList, graphColors, maxDegree, start, end);

				if (satDegree > max){
					max = satDegree;
					index = i;				
				}

				if (satDegree == max){
					if (degree(i,degreeList) > degree(index,degreeList))
						index = i;
				}
			}

			numColored = color(index, compactAdjacencyList, vertexStartList, graphColors, maxDegree, numColored, start, end);
		}
	}
}

//----------------------- First Fit Adjacency List -----------------------//
//
// Author: Pascal
// First Fit
__global__ void colorGraph_FF(long *adjacencyListD, long *colors, long size, long maxDegree){
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
		
		if (colors[i] > numColors)
			numColors = colors[i];		
	}
}



__global__ void colorGraph_FF(long *compactAdjacencyListD, long *vertexStartListD, long *colors, long size, long maxDegree){
	int i, j, start, end;
	int subGraphSize, numColors = 0;
	
	subGraphSize = size/(gridDim.x * blockDim.x);
	start = (size/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
	end = start + subGraphSize;
	

	long degreeArray[100];
	for(i=start; i<end; i++)
	{
		for(j=0; j<maxDegree; j++)
			degreeArray[j] = j+1;


		// check the colors  
		for (j=vertexStartListD[i]; j<vertexStartListD[i+1]; j++){
			if (i == j)  
				continue;  
			
			if (colors[ compactAdjacencyListD[j] ] != 0)  
				degreeArray[colors[   compactAdjacencyListD[j]   ]-1] = 0;   // set connected spots to 0  
		}  

		

		for(j=0; j<maxDegree; j++)
			if(degreeArray[j] != 0){
				colors[i] = degreeArray[j];
				break;
			}
		
		if (colors[i] > numColors)
			numColors = colors[i];		
	}
}


//----------------------- Detects conflicts -----------------------//
//
// Author: Peihong
__global__ void conflictsDetection(long *adjacentListD, long *boundaryListD, long *colors, long *conflictD, long size, long boundarySize, long maxDegree){
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j;
	if(idx < boundarySize){
		i = boundaryListD[idx];
		conflictD[idx] = 0;
		for(long k= 0; k < maxDegree; k++)
		{
			j = adjacentListD[i*maxDegree + k];
			if(j>i && (colors[i] == colors[j]))
			{
				//conflictD[idx] = min(i,j)+1;	
				conflictD[idx] = i+1;	
			}		
		}
	}
}


__global__ void conflictsDetection(long *compactAdjacencyList, long *vertexStartList, long *boundaryListD, long *colors, long *conflictD, long size, long boundarySize, long maxDegree){
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	long i;

	if (idx < boundarySize){
		i = boundaryListD[idx];
		conflictD[idx] = 0;

		for (long j=vertexStartList[i]; j<vertexStartList[i+1]; j++){
			long vertex = compactAdjacencyList[j];

			if ( (vertex > i) && (colors[i] == colors[vertex]))
				conflictD[idx] = i+1;
		} 

	}
}


//----------------------- Main -----------------------//

extern "C"
void cudaGraphColoring(long *adjacentList, long *compactAdjacencyList, long *vertexStartList, long *boundaryList, long *graphColors, long *degreeList, long *conflict, long boundarySize, long maxDegree, long graphSize, long numEdges)
{
	long *adjacentListD, *colorsD, *conflictD, *boundaryListD, *degreeListD, *vertexStartListD, *compactAdjacencyListD;     
	long gridsize = ceil((float)boundarySize/(float)SUBSIZE_BOUNDARY);
	long blocksize = SUBSIZE_BOUNDARY;
	
	cudaEvent_t start_col, start_confl, stop_col, stop_confl, start_mem, stop_mem;         
    float elapsedTime_memory, elapsedTime_col, elapsedTime_confl; 



//-------------- memory transfer -----------------!
	cudaEventCreate(&start_mem); 
    cudaEventCreate(&stop_mem); 
    cudaEventRecord(start_mem, 0); 
	

	cudaMalloc((void**)&adjacentListD, graphSize*maxDegree*sizeof(long));
	cudaMalloc((void**)&colorsD, graphSize*sizeof(long));
	cudaMalloc((void**)&conflictD, boundarySize*sizeof(long));
	cudaMalloc((void**)&boundaryListD, boundarySize*sizeof(long));
	cudaMalloc((void**)&degreeListD, graphSize*sizeof(long));

	cudaMalloc((void**)&compactAdjacencyListD, (numEdges*2)*sizeof(long));
	cudaMalloc((void**)&vertexStartListD, (graphSize+1)*sizeof(long));
	

	cudaMemcpy(adjacentListD, adjacentList, graphSize*maxDegree*sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(colorsD, graphColors, graphSize*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(boundaryListD, boundaryList, boundarySize*sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(degreeListD, degreeList, graphSize*sizeof(long), cudaMemcpyHostToDevice);

	cudaMemcpy(vertexStartListD, vertexStartList, (graphSize+1)*sizeof(long), cudaMemcpyHostToDevice);
	cudaMemcpy(compactAdjacencyListD, compactAdjacencyList, (numEdges*2)*sizeof(long), cudaMemcpyHostToDevice);
	
	
	cudaEventRecord(stop_mem, 0); 
    cudaEventSynchronize(stop_mem); 
	
	


	dim3 dimGrid_col(GRIDSIZE);
	dim3 dimBlock_col(BLOCKSIZE);
	
	dim3 dimGrid_confl(gridsize);
	dim3 dimBlock_confl(blocksize);
	
	


//-------------- Sequential Graph coloring -----------------!
	cudaEventCreate(&start_col); 
    cudaEventCreate(&stop_col); 
    cudaEventRecord(start_col, 0); 
	
	
	//colorGraph_FF<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, graphSize, maxDegree);				// First Fit
	//colorGraph_FF<<<dimGrid_col, dimBlock_col>>>(compactAdjacencyListD, vertexStartListD, colorsD, graphSize, maxDegree);				// First Fit

	//colorGraph_SDO<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, degreeListD,graphSize, maxDegree);		// SDO improved
	colorGraph_SDO<<<dimGrid_col, dimBlock_col>>>(compactAdjacencyListD, vertexStartListD, colorsD, degreeListD,graphSize, maxDegree);		// SDO improved
	
	
	cudaEventRecord(stop_col, 0); 
    cudaEventSynchronize(stop_col); 

	



//-------------- Conflict resolution -----------------!
	cudaEventCreate(&start_confl); 
    cudaEventCreate(&stop_confl); 
    cudaEventRecord(start_confl, 0); 
	
	//conflictsDetection<<<dimGrid_confl, dimBlock_confl>>>(adjacentListD, boundaryListD, colorsD, conflictD, graphSize, boundarySize, maxDegree);
	conflictsDetection<<<dimGrid_confl, dimBlock_confl>>>(compactAdjacencyListD, vertexStartListD, boundaryListD, colorsD, conflictD, graphSize, boundarySize, maxDegree);
	

	cudaEventRecord(stop_confl, 0); 
    cudaEventSynchronize(stop_confl); 
	




//-------------- Cleanup -----------------!
	cudaMemcpy(graphColors, colorsD, graphSize*sizeof(long), cudaMemcpyDeviceToHost);
	cudaMemcpy(conflict, conflictD, boundarySize*sizeof(long), cudaMemcpyDeviceToHost);



	cudaEventElapsedTime(&elapsedTime_memory, start_mem, stop_mem); 
	cudaEventElapsedTime(&elapsedTime_col, start_col, stop_col); 
	cudaEventElapsedTime(&elapsedTime_confl, start_confl, stop_confl); 

	cout << "GPU timings ~ Memory transfer: " << elapsedTime_memory  << " ms     Coloring: " 
		 << elapsedTime_col << " ms    Conflict: " << elapsedTime_confl << " ms" << endl; 


	cudaFree(adjacentListD);
	cudaFree(colorsD);
	cudaFree(conflictD);
	cudaFree(boundaryListD);
}

