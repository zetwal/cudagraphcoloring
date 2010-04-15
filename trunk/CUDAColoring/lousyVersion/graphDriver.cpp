// Graph coloring
 
#include <ctime>
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <iostream>
#include <math.h> 
#include <set>
#include "graphColoring.h"

using namespace std; 

#define UNIQUE_CONFLICT



// Author: Pascal
// genetates a graph
void generateMatrix(int *matrix, int size, int num){ 
	int x, y; 
	int count = 0; 

	//cout<<"num="<<num<<endl; 
	//cout<<"enter generateMatrix()"<<endl;
	
	while (count < num){ 
		x = rand()%GRAPHSIZE; 
		y = rand()%GRAPHSIZE; 
		//cout<<"x="<<x<<endl;
		//cout<<"y="<<y<<endl;

		if (matrix[x*size + y] != 1)		// if not already assigned an edge
			if (x != y){ 
				matrix[x*size + y] = 1;       // non directional graph 
				matrix[y*size + x] = 1; 
			count++; 
			} 
	} 
} 



// Author: Pascal
// get the degree information for a graph
int getMaxDegree(int *adjacencyMatrix, int size){ 
	int maxDegree = 0;  
	int degree; 
	
	for (int i=0; i<size; i++){ 
		degree = 0; 
		
		for (int j=0; j<size; j++)          
			if (    adjacencyMatrix[i*size + j] == 1) 
				degree++; 
		
		if (degree > maxDegree) 
			maxDegree = degree; 
	} 
	
	return maxDegree; 
} 




// Author: Pascal
// colors a graph: First Fit algo used
int colorGraph(int *matrix, int *colors, int size, int maxDegree){ 
	int numColors = 0; 
	int i, j; 
	
	int * degreeArray; 
	degreeArray = new int[maxDegree+1]; 
	
	
	for (i=0; i<size; i++) 
	{                
		// initialize degree array 
		for (j=0; j<=maxDegree; j++) 
			degreeArray[j] = j+1; 
		
		
		// check the colors 
		for (j=0; j<size; j++){ 
			if (i == j) 
				continue; 
			
			// check connected 
			if (    matrix[i*size + j] == 1) 
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
	delete[] degreeArray;
	
	return numColors; 
} 




int min(int n1, int n2)
{
    if(n1>=n2)
		return n2;
    else
		return n1;
}



// Author: Peihong & Shusen
// get the conflicts in the initial coloring
int getConflicts(int *adjacencyMatrix, int *graphColors, int *conflict)
{	
	int numSub = GRIDSIZE*BLOCKSIZE;
	int i,j,n;

	set<int> conflictSet;
        conflictSet.clear();

        
        int conflictCount = 0;
        for(n=0; n<numSub; n++)
        {
                int size;
                if(n < numSub-1)
                {
                        size = SUBSIZE;
                }
                else
                {
                        size = GRAPHSIZE - (numSub-1)*SUBSIZE;
                }
                
                for(i = 0; i < size; i++)
                {
                        int ii = i + n*SUBSIZE;
                        for(j = (ii+1); j < GRAPHSIZE; j++)
                        {
                                if(j > n*SUBSIZE && j < (n+1)*SUBSIZE)
                                {
                                        continue;
                                }
                                
                                if( adjacencyMatrix[ii*GRAPHSIZE + j] == 1 && (graphColors[ii] == graphColors[j]))
                                {               
                                        conflictSet.insert(min(ii,j) + 1);
                                }
                        }               
                }
                
        }
        
    
        
        set<int>::iterator it = conflictSet.begin();
        for (int i=0; it != conflictSet.end(); it++) 
        {
                conflict[i] = *it; 
                i++;
        }
     

        return conflictSet.size();
}



// Author: Shusen
// Solves conflicts
int solveConflict(int *matrix, int size, int *conflict, int conflictSize, int *graphColors)
{
	set<int> localColor;
	set<int> globalColor;
	set<int>::iterator it;
	int colorCount = 0;
	
	for(int i = 0; i<size; i++)
		globalColor.insert(graphColors[i]);
	
	//go over all the conflicts
	
	for(int i=0; i<conflictSize; i++)
	{
		localColor.clear();
		int nodeIndex = conflict[i]-1;  //index begin with zero
		int currentColor = graphColors[nodeIndex];
		//check all the adjacency node
		for(int j = 0; j < size; j++)
			if(matrix[nodeIndex*size + j] ==1) //adjacency to current node
				localColor.insert(graphColors[j]);
		
		//recolor the current conflict based on the adjacency color stored in localColor
		
		//get coloring = globalColor - localColor
		set<int> coloring = globalColor;
		it = localColor.begin();
		for(; it != localColor.end(); it++)
			coloring.erase(*it);
		
		if(coloring.size() != 0)
		{
			it = coloring.begin();
			graphColors[nodeIndex] = *it;
		}
		else
		{
			globalColor.insert(globalColor.size() + 1);
			graphColors[nodeIndex] = globalColor.size();
		}
		
	}

	return globalColor.size();
}



// Author: Pascal
// Checks if a graph has conflicts or not
void checkCorrectColoring(int *adjacencyMatrix, int *graphColors){
	int numErrors = 0;
	
	cout << endl << "==================" << endl << "Error checking for Graph" << endl;
	
	for (int i=0; i<GRAPHSIZE; i++)			// we check each row
	{
		int nodeColor = graphColors[i];
		int numErrorsOnRow = 0;
		
		for (int j=0; j<GRAPHSIZE;j++){	// check each column in the matrix
			
			// skip itself
			if (i == j)
				continue;
			
			if (adjacencyMatrix[i*GRAPHSIZE + j] == 1)	// there is a connection to that node
				if (graphColors[j] == nodeColor)
				{
					cout << "Color collision from: " << i << " @ " << nodeColor << "  to: " << j << " @ " << graphColors[j] << endl;
					numErrors++;
					numErrorsOnRow++;
				}
		}
		
		if (numErrorsOnRow != 0)
			cout << "Errors for node " << i << " : " << numErrorsOnRow << endl;
	}
	
	cout << "Color errors for graph : " << numErrors << endl << "==================== " << endl ;	
}





int main(){ 
	int *adjacencyMatrix = new int[GRAPHSIZE*GRAPHSIZE*sizeof(int)]; 
	int *graphColors = new int[GRAPHSIZE*sizeof(int)];         


	memset(adjacencyMatrix, 0, GRAPHSIZE*GRAPHSIZE*sizeof(int));
	memset(graphColors, 0, GRAPHSIZE*sizeof(int));

	
	int numColorsSeq, numColorsParallel;
	int maxDegree; 
	numColorsSeq = numColorsParallel = 0;
	float GPUtime;

	srand ( time(NULL) );  // initialize random numbers   


	cudaEvent_t start, stop, stop_1, stop_2, stop_3, stop_4;	
	float elapsedTimeCPU, elapsedTimeGPU, elapsedTimeGPU_1, elapsedTimeGPU_2, elapsedTimeGPU_3, elapsedTimeGPU_4;
	
                                                       
	

//------------- Graph Creation --------------//
// initialize graph 
	generateMatrix(adjacencyMatrix, GRAPHSIZE, NUMEDGES); 

	
// Display graph
/** 
	for (int i=0; i<GRAPHSIZE; i++){ 
		for (int j=0; j<GRAPHSIZE; j++) 
			cout << adjacencyMatrix[i*GRAPHSIZE + j] << "  "; 
		
		cout << endl; 
	} 
/**/	
	
// determining the maximum degree 
	maxDegree = getMaxDegree(adjacencyMatrix, GRAPHSIZE); 
	
	cout << "Max degree: " << maxDegree << endl; 
	
	


//------------- Sequential Graph Coloring --------------//


	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 

	numColorsSeq = colorGraph(adjacencyMatrix, graphColors, GRAPHSIZE, maxDegree);     

	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&elapsedTimeCPU, start, stop);
	
   
	cout<<"global graph coloring results:"<<endl;
	for (int k=0; k<GRAPHSIZE; k++) 
		cout << graphColors[k] << "  "; 
	cout << endl; 
	



//------------- Checking for color conflict --------------//
	checkCorrectColoring(adjacencyMatrix, graphColors);
	cout << endl; 
	
	



//------------- Parallel Graph Coloring --------------//

	int *conflict = new int[GRAPHSIZE*sizeof(int)];				// conflict array
	memset(conflict, 0, GRAPHSIZE*sizeof(int));					// conflict array initialized to 0 

	memset(graphColors, 0, GRAPHSIZE*sizeof(int));  				// reset colors to 0


	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&stop_2);
	cudaEventCreate(&stop_3);
	cudaEventCreate(&stop_4);
	cudaEventRecord(start, 0);



// Steps 1 & 2: partition and color
	subGraphColoring(adjacencyMatrix, graphColors, maxDegree);					// subgraph coloring @ CUDA

	cudaEventRecord(stop_1, 0);	cudaEventSynchronize(stop_1);	
	cudaEventRecord(stop_2, 0);	cudaEventSynchronize(stop_2);	



// Step 3:	get conflicts
	int conflictCount = getConflicts(adjacencyMatrix, graphColors, conflict);	
	
	cudaEventRecord(stop_3, 0);	cudaEventSynchronize(stop_3);



// Step 4: solve conflicts
    numColorsParallel = solveConflict(adjacencyMatrix,  GRAPHSIZE, conflict, conflictCount, graphColors);

	

	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
	cudaEventElapsedTime(&elapsedTimeGPU_1, start, stop_1);
	cudaEventElapsedTime(&elapsedTimeGPU_2, start, stop_2);
	cudaEventElapsedTime(&elapsedTimeGPU_3, start, stop_3);


// Display information
	cout << "List of conflicting nodes:"<<endl;
	for (int k=0; k<conflictCount; k++) 
		cout << conflict[k] << "  "; 
	cout << endl << endl; 
		

	cout << "Global graph coloring results:"<<endl;
	for (int k=0; k<GRAPHSIZE; k++) 
		cout << graphColors[k] << "  "; 
	cout << endl << endl;
	

       
	//cout << "GPU Partitioning and coloring time: " << GPUtime << " ms" << endl;
	cout << "CPU time: " << elapsedTimeCPU << " ms    - GPU Time: " << elapsedTimeGPU << " ms" << endl;
	cout << "ALGO step 1: " << elapsedTimeGPU_1 << " ms" << endl;
	cout << "ALGO step 2: " << elapsedTimeGPU_2 << " ms" << endl;
	cout << "ALGO step 3: " << elapsedTimeGPU_3 - elapsedTimeGPU_2 << " ms" << endl;
	cout << "ALGO step 4: " << elapsedTimeGPU - elapsedTimeGPU_3 << " ms" << endl;
	cout << endl << "Sequential Colors: " << numColorsSeq << " 	- 	Prarallel Colors: " << numColorsParallel << endl;    
	

//------------- Checking for color conflict --------------//
	checkCorrectColoring(adjacencyMatrix, graphColors);
	


//------------- Cleanup --------------//
	delete[] adjacencyMatrix;
	delete[] graphColors;
	delete[] conflict;
	
	return 0; 
} 
