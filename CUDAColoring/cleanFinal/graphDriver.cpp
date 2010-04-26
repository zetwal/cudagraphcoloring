// Graph coloring 

#include <ctime> 
#include <stdio.h>  
#include <stdlib.h>  
#include <time.h>  
#include <iostream> 
#include <math.h>  
#include <set> 
#include "graphColoring.h" 
#include <assert.h>

#include <fstream>
using namespace std;  

//----------------------- Read Sparse Graph from Matrix Market format --------------//
// Author: Shusen
void inline swap(long &a, long &b) {long temp = a; a=b; b=temp; }
long inline findmax(long *a, long n){
	long max=0;
	for(long i=0; i<n; i++)
		if(a[i]>max)
			max = a[i];
	return max;	
}

long getAdjacentListFromSparseMartix_mtx(long *&adjacencyList, long &graphsize, const char* filename) //return the maxDegree
{
	long maxDegree=0;
	long row=0, col=0, entries=0;

	ifstream mtxf;
	mtxf.open(filename);
	cout << string(filename) << endl;

	mtxf >> row >> col >> entries ;
	cout<< row <<" " << col <<" " << entries << endl;
	graphsize = col>row? col:row;

	//calculate maxDegree in the following loop
	long donotcare = 0;
	long nodecol = 0;
	long noderow = 0;

	long *graphsizeArray = new long[graphsize];
	memset(graphsizeArray, 0 , sizeof(long)*graphsize);
	long j=0;
	for(long i=0; i<entries; i++)
	{
		j++;
		mtxf >> noderow >> nodecol >> donotcare;
		//cout << noderow << " " << nodecol << " " << donotcare << endl;
		//assert(noderow!=nodecol);
		if(noderow == nodecol)
			continue;
		graphsizeArray[noderow-1]++;
		graphsizeArray[nodecol-1]++;
	}
	maxDegree = findmax(graphsizeArray,graphsize);



	cout << j <<" maxDegree: "<< maxDegree << endl;
	mtxf.close();

	//create the adjacency list matrix
	
	adjacencyList = new long[maxDegree*graphsize];
	long *adjacencyListLength = new long[graphsize];
	memset(adjacencyList, -1, sizeof(long)*maxDegree*graphsize);
	memset(adjacencyListLength, 0, sizeof(long)*graphsize); //indicate how many element has been stored in the list of each node
	//for(long i=0; i<maxDegree; i++)
	//	cout<<adjacencyList[i]<<endl;
	


	//update the adjacency list matrix from the file
	mtxf.open(filename);
	long nodeindex=0, connection=0;
	
	mtxf >> donotcare >> donotcare >> donotcare;
	//cout << donotcare << endl;

	for(long i=0; i<entries; i++)
	{
		mtxf >> connection >> nodeindex >> donotcare;
		if(noderow == nodecol)
			continue;
		//because node in mtx file begin with 1 but in the program begin with 0
		adjacencyList[(nodeindex-1)*maxDegree + adjacencyListLength[nodeindex-1] ]=connection-1; 
		adjacencyListLength[nodeindex-1]++;

		swap(connection, nodeindex);
		adjacencyList[(nodeindex-1)*maxDegree + adjacencyListLength[nodeindex-1] ]=connection-1; 
		adjacencyListLength[nodeindex-1]++;		
	}

	delete [] adjacencyListLength;

	return maxDegree;

}





//----------------------- Graph initializations -----------------------//

// Author: Pascal 
// genetates a graph 
void generateMatrix(long *matrix, long numEdges, long graphSize){  
	long x, y;  
	long count = 0;  
	
	while (count < numEdges){  
		x = rand()%graphSize;  
		y = rand()%graphSize;  
		
		if (matrix[x*graphSize + y] != 1)            // if not already assigned an edge 
			if (x != y){  
				matrix[x*graphSize + y] = 1;       // non directional graph  
				matrix[y*graphSize + x] = 1;  
				count++;  
			}  
	}  
}  




// Author:Peihong
// node index start from 1
// gets an adjacency list from an adjacencyMatrix
void getAdjacentList(long *adjacencyMatrix, long *adjacentList, long size, long maxDegree)
{
	for (long i=0; i<size; i++){ 
		long nbCount = 0;
		for (long j=0; j<size; j++){
			if ( adjacencyMatrix[i*size + j] == 1)  
			{
				adjacentList[i*maxDegree + nbCount] = j;
				nbCount++;
			}
		}
	}

	/*
	// Adj list display
	for (long i=0; i<10; i++){
		for (long j=0; j<maxDegree; j++){
			cout << adjacentList[i*maxDegree + j] << " ";
		}
		cout << endl;
	}
	*/
}



void getCompactAdjacencyList(long *adjacencyList, long *compactAdjacencyList, long *vertexStartList, long size, long maxDegree){
	long count;

	for (long i=0; i<size; i++){
		if (i == 0)
			vertexStartList[i] = 0;
		else
			vertexStartList[i] = vertexStartList[i-1] + count;

		count = 0;
		for (long j=0; j<maxDegree; j++)
			if 	(adjacencyList[i*maxDegree +j] != -1)
			{
				compactAdjacencyList[ vertexStartList[i] + j ] = adjacencyList[i*maxDegree +j];
				count++;
			}
			else
				break;	
	}
}




// Author: Pascal 
// get the degree information for a graph 
long getMaxDegree(long *adjacencyMatrix, long size){  
	long maxDegree = 0;   
	long degree;  
	
	for (long i=0; i<size; i++){  
		degree = 0;  
		
		for (long j=0; j<size; j++)           
			if (    adjacencyMatrix[i*size + j] == 1)  
				degree++;  
		
		if (degree > maxDegree)  
			maxDegree = degree;  
	}  
	
	return maxDegree;  
}  




// Author: Pascal
// get the degree of each element in the graph and returns the maximum degree
void getDegreeList(long *adjacencyList, long *degreeList, long sizeGraph, long maxDegree){
    for (long i=0; i<sizeGraph; i++){
        long count = 0;
        
        for (long j=0; j<maxDegree; j++){
            if (adjacencyList[i*maxDegree + j] != -1)
                count++;
            else
                break;  
        }

        degreeList[i] = count;
    }
}




long inline min(long n1, long n2) 
{ 
	if (n1>=n2) 
		return n2; 
	else 
		return n1; 
} 

// Author: Peihong
long getBoundaryList(long *adjacencyMatrix, long *boundaryList, long size, long &boundaryCount){  
	long maxDegree = 0;   
	long degree;  
	long subSize = size/(GRIDSIZE*BLOCKSIZE);
	
	
	set<long> boundarySet; 
	boundarySet.clear(); 


	
	for (long i=0; i<size; i++){  
		degree = 0;  

		long subIdx = i/(float)subSize;
		long start = subIdx * subSize;
		long end = min( (subIdx + 1)*subSize, size );

		for (long j=0; j<size; j++){           
			if ( adjacencyMatrix[i*size + j] == 1)  
				degree++;  
			if ( adjacencyMatrix[i*size + j] == 1 && (j < start || j >= end))
			{
				boundarySet.insert(i);
			}
			
		}
		
		if (degree > maxDegree)  
			maxDegree = degree;  	
	} 
	
	boundaryCount = boundarySet.size();

	set<long>::iterator it = boundarySet.begin(); 
	for (long i=0; it != boundarySet.end(); it++)  
	{ 
		boundaryList[i] = *it;
		i++; 
	}  
	
	return maxDegree;  
}  




// Author: Peihong
void getBoundaryList(long *compactAdjacencyList, long *vertexStartList, long *boundaryList, long size, long &boundaryCount){    
	long degree;  
	long subSize = size/(GRIDSIZE*BLOCKSIZE);
	
	set<long> boundarySet; 
	boundarySet.clear(); 

	for (long i=0; i<size; i++){  
		degree = 0;  

		long subIdx = i/(float)subSize;
		long start = subIdx * subSize;
		long end = min( (subIdx + 1)*subSize, size );


		for (long j=vertexStartList[i]; j<vertexStartList[i+1]; j++){
			if ((compactAdjacencyList[j] < start) || (compactAdjacencyList[j] >= end))
				 boundarySet.insert(i);
		}  
	} 
	
	boundaryCount = boundarySet.size();

	set<long>::iterator it = boundarySet.begin(); 
	for (long i=0; it != boundarySet.end(); it++)  
	{ 
		boundaryList[i] = *it;
		i++; 
	}  
}  



// Author: Peihong & Shusen
// getBoundaryList from adjacency list representation
void getBoundaryList_adjList(long *adjacencyList, long *boundaryList, long size, long maxDegree, long &boundaryCount){  	
	set<long> boundarySet; 
	boundarySet.clear(); 
	long subSize = size/(GRIDSIZE*BLOCKSIZE);

	assert(adjacencyList);
	
	for (long i=0; i<size; i++){  


		long subIdx = i/(float)subSize;
		long start = subIdx * subSize;
		long end = min( (subIdx + 1)*subSize, size );

		for (long j=0; j<maxDegree; j++){   
			assert(i*maxDegree+j <= (size-1)*maxDegree+maxDegree-1);        

			if ( adjacencyList[i*maxDegree + j] < start || adjacencyList[i*maxDegree + j] >= end)
			{
				boundarySet.insert(i);
			}
			
		}

	} 

	
	boundaryCount = boundarySet.size();
	boundaryList = new long[boundaryCount];


 
	set<long>::iterator it = boundarySet.begin(); 
	for (long i=0; it != boundarySet.end(); it++)  
	{ 
		boundaryList[i] = *it;
		i++; 
	}  
//cout << "Debug:" <<endl;  
}  





//----------------------- Fast Fit Graph Coloring -----------------------//

// Author: Pascal
// Based on the compact adjacency list representation
void colorGraph_FF(long *compactAdjacencyList, long *vertexStartList, long *colors, long size, long maxDegree){  
	long i, j;  
	
	long * degreeArray;  
	degreeArray = new long[maxDegree+1];  
	
	
	for (i=0; i<size; i++)  
	{                 
		// initialize degree array  
		for (j=0; j<=maxDegree; j++)  
			degreeArray[j] = j+1;  
		
		
		// check the colors  
		for (j=vertexStartList[i]; j<vertexStartList[i+1]; j++){
			if (i == j)  
				continue;  
			
			// check connected  
			if (colors[ compactAdjacencyList[j] ] != 0)  
				degreeArray[colors[   compactAdjacencyList[j]   ]-1] = 0;   // set connected spots to 0  
		}  
		
		for (j=0; j<=maxDegree; j++)  
			if (degreeArray[j] != 0){  
				colors[i] = degreeArray[j];  
				break;  
			}   
	}  

	delete[] degreeArray; 
}  


//-----------------------------------------------------------

// Author: Pascal & Shusen
// GraphColor Adjacency list
long colorGraph_FF(long *list, long *colors, long size, long maxDegree){  
	long numColors = 0;  
	long i, j;  
	
	long * degreeArray;  
	degreeArray = new long[maxDegree+1];  
	
	
	for (i=0; i<size; i++)  
	{                 
		// initialize degree array  
		for (j=0; j<=maxDegree; j++)  
			degreeArray[j] = j+1;  
		
		
		// check the colors  
		for (j=0; j<maxDegree; j++){  
			if (i == j)  
				continue;  
			
			// check connected  
			if (    list[i*maxDegree + j] != -1)  
				if (colors[list[i*maxDegree + j]] != 0)  
					degreeArray[colors[list[i*maxDegree + j]]-1] = 0;   // set connected spots to 0  
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




//----------------------- SDO Improved Graph Coloring -----------------------//

// Author: Pascal
// returns the degree of that node
long degree(long vertex, long *degreeList){
        return degreeList[vertex];
}



// Author: Pascal
// return the saturation of that node
long saturation(long vertex, long *adjacencyList, long *graphColors, long maxDegree){
    long saturation = 0;
    long *colors = new long[maxDegree+1];

    memset(colors, 0, (maxDegree+1)*sizeof(long));           // initialize array


    for (long i=0; i<maxDegree; i++){
        if (adjacencyList[vertex*maxDegree + i] != -1)
            colors[ graphColors[i] ] = 1;                      // at each colored set the array to 1
        else
            break;
    }


    for (long i=1; i<maxDegree+1; i++)                                       // count the number of 1's but skip uncolored
        if (colors[i] == 1)
            saturation++;

	delete[] colors; 	

    return saturation;
}




// Author: Pascal
// colors the vertex with the min possible color
long color(long vertex, long *adjacencyList, long *graphColors, long maxDegree, long numColored){
    long *colors = new long[maxDegree + 1];
    memset(colors, 0, (maxDegree+1)*sizeof(long));   
    
    if (graphColors[vertex] == 0)
            numColored++;
    
    for (long i=0; i<maxDegree; i++)                                         // set the index of the color to 1
        if (adjacencyList[vertex*maxDegree + i] != -1)
            colors[  graphColors[  adjacencyList[vertex*maxDegree + i]  ]  ] = 1;
        else {
            break;
        }

    

    for (long i=1; i<maxDegree+1; i++)                                       // nodes still equal to 0 are unassigned
        if (colors[i] != 1){
            graphColors[vertex] = i;
            break;
        }
    
	delete[] colors; 

    return numColored;
}




// Author: Pascal
long sdoIm(long *adjacencyList, long *graphColors, long *degreeList, long sizeGraph, long maxDegree){
    long satDegree, numColored, max, index;
    numColored = 0;
    long iterations = 0;
    

    while (numColored < sizeGraph){
        max = -1;
        
        for (long i=0; i<sizeGraph; i++){
            if (graphColors[i] == 0)                        // not colored
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
            iterations++;
        }
    }
    
    return iterations;
}

//-----------------------------------------------------------------




// Author: Pascal
// colors the vertex with the min possible color
long color(long vertex, long *compactAdjacencyList, long *vertexStartList, long *graphColors, long maxDegree, long numColored){
    long *colors = new long[maxDegree + 1];
    memset(colors, 0, (maxDegree+1)*sizeof(long));   
    
    if (graphColors[vertex] == 0)
            numColored++;

	
	for (long i=vertexStartList[vertex]; i<vertexStartList[vertex+1]; i++){
		colors[ graphColors[ compactAdjacencyList[i] ] ] = 1;
	}
	

    for (long i=1; i<maxDegree+1; i++)                                       // nodes still equal to 0 are unassigned
        if (colors[i] != 1){
            graphColors[vertex] = i;
            break;
        }
    
	delete[] colors; 

    return numColored;
}


// Author: Pascal
// return the saturation of that node
long saturation(long vertex, long *compactAdjacencyList, long *vertexStartList, long *graphColors, long maxDegree){
    long saturation = 0;
    long *colors = new long[maxDegree+1];

    memset(colors, 0, (maxDegree+1)*sizeof(long));           // initialize array



	for (long i=vertexStartList[vertex]; i<vertexStartList[vertex+1]; i++){
		colors[ graphColors[ compactAdjacencyList[i] ] ] = 1;
	}  


    for (long i=1; i<maxDegree+1; i++)                                       // count the number of 1's but skip uncolored
        if (colors[i] == 1)
            saturation++;

	delete[] colors; 	

    return saturation;
}


// Author: Pascal
long sdoIm(long *compactAdjacencyList, long *vertexStartList, long *graphColors, long *degreeList, long sizeGraph, long maxDegree){
    long satDegree, numColored, max, index;
    numColored = 0;
    long iterations = 0;
    

    while (numColored < sizeGraph){
        max = -1;
        
        for (long i=0; i<sizeGraph; i++){
            if (graphColors[i] == 0)                        // not colored
            {
                satDegree = saturation(i, compactAdjacencyList, vertexStartList, graphColors, maxDegree);

                if (satDegree > max){
                    max = satDegree;
                    index = i;                              
                }

                if (satDegree == max){
                    if (degree(i,degreeList) > degree(index,degreeList))
                        index = i;
                }
            }
			
            numColored = color(index, compactAdjacencyList, vertexStartList, graphColors, maxDegree, numColored);
            iterations++;
        }
    }
    
    return iterations;
}



//----------------------- Conflict Solve -----------------------//

void conflictSolveSDO(long *adjacencyList, long *conflict, long conflictSize, long *graphColors, long *degreeList, long sizeGraph, long maxDegree){
    long satDegree, numColored, max, index;
    numColored = 0;
   
	// Set their color to 0
	for (long i=0; i<conflictSize; i++)
		graphColors[conflict[i]-1] = 0;
    

    while (numColored < conflictSize){
        max = -1;
        
        for (long i=0; i<conflictSize; i++){
			long vertex = conflict[i]-1;
            if (graphColors[vertex] == 0)                        // not colored
            {
                satDegree = saturation(vertex, adjacencyList, graphColors, maxDegree);

                if (satDegree > max){
                    max = satDegree;
                    index = vertex;                              
                }

                if (satDegree == max){
                    if (degree(vertex,degreeList) > degree(index,degreeList))
                        index = vertex;
                }
            }

            numColored = color(index,adjacencyList,graphColors, maxDegree, numColored);
        }
    }
}


void conflictSolveSDO(long *compactAdjacencyList, long *vertexStartList, long *conflict, long conflictSize, long *graphColors, long *degreeList, long sizeGraph, long maxDegree){
    long satDegree, numColored, max, index;
    numColored = 0;
   
	// Set their color to 0
	for (long i=0; i<conflictSize; i++)
		graphColors[conflict[i]-1] = 0;
    

    while (numColored < conflictSize){
        max = -1;
        
        for (long i=0; i<conflictSize; i++){
			long vertex = conflict[i]-1;
            if (graphColors[vertex] == 0)                        // not colored
            {
                satDegree = saturation(vertex, compactAdjacencyList, vertexStartList, graphColors, maxDegree);

                if (satDegree > max){
                    max = satDegree;
                    index = vertex;                              
                }

                if (satDegree == max){
                    if (degree(vertex,degreeList) > degree(index,degreeList))
                        index = vertex;
                }
            }

            numColored = color(index, compactAdjacencyList, vertexStartList, graphColors, maxDegree, numColored);
        }
    }
}

//-------------------------------------------------------------

// Author: Pascal & Shusen
// Solves conflicts using Fast Fit
void conflictSolveFF(long *Adjlist, long size, long *conflict, long conflictSize, long *graphColors, long maxDegree){
	long i, j, vertex, *colorList, *setColors;
	colorList = new long[maxDegree];
	setColors = new long[maxDegree];


	// assign colors up to maxDegree in setColors
	for (i=0; i<maxDegree; i++){
	        setColors[i] = i+1;
	}


	for (i=0; i<conflictSize; i++){
        memcpy(colorList, setColors, maxDegree*sizeof(long));                    // set the colors in colorList to be same as setColors

        vertex = conflict[i]-1;
        

        for (j=0; j<maxDegree; j++){                                            // cycle through the graph
        		if ( Adjlist[vertex*maxDegree + j] != -1 )                      		//      check if node is connected
				colorList[ graphColors[Adjlist[vertex*maxDegree + j]]-1 ] = 0;
			else 
                break;    
        }


        for (j=0; j<maxDegree; j++){                                       	// check the colorList array
        		if (colorList[j] != 0){                                         //    at the first spot where we have a color not assigned
            		graphColors[vertex] = colorList[j];                         //       we assign that color to the node and
                break;                                                      //   	 exit to the next
            }
        }

	}
}



void conflictSolveFF(long *compactAdjacencyList, long *vertexStartList, long size, long *conflict, long conflictSize, long *graphColors, long maxDegree){
	long i, j, vertex, *colorList, *setColors;
	colorList = new long[maxDegree];
	setColors = new long[maxDegree];


	// assign colors up to maxDegree in setColors
	for (i=0; i<maxDegree; i++){
	        setColors[i] = i+1;
	}


	for (i=0; i<conflictSize; i++){
        memcpy(colorList, setColors, maxDegree*sizeof(long));                    // set the colors in colorList to be same as setColors

        vertex = conflict[i]-1;
        

        for (j=vertexStartList[vertex]; j<vertexStartList[vertex+1]; j++){   	
			colorList[ graphColors[   compactAdjacencyList[j]    ]-1 ] = 0;  
        }


        for (j=0; j<maxDegree; j++){                                       	// check the colorList array
        		if (colorList[j] != 0){                                         //    at the first spot where we have a color not assigned
            		graphColors[vertex] = colorList[j];                         //       we assign that color to the node and
                break;                                                      //   	 exit to the next
            }
        }

	}
}


//----------------------- Checking for error -----------------------//

// Author: Pascal 
// Checks if a graph has conflicts or not 
void checkCorrectColoring(long *adjacencyMatrix, long *graphColors, long graphSize){ 
	long numErrors = 0; 
	
	cout << endl << "==================" << endl << "Error checking for Graph" << endl; 
	
	for (long i=0; i<graphSize; i++)                 // we check each row 
	{ 
		long nodeColor = graphColors[i]; 
		long numErrorsOnRow = 0; 
		
		for (long j=0; j<graphSize;j++){ // check each column in the matrix 
			
			// skip itself 
			if (i == j) 
				continue; 
			
			if (adjacencyMatrix[i*graphSize + j] == 1)      // there is a connection to that node 
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





void checkCorrectColoring(long *compactAdjacencyList, long *vertexStartList, long *graphColors, long graphSize){ 
	int numErrors = 0; 
	
	cout << endl << "==================" << endl << "Error checking for Graph" << endl; 
	
	for (long i=0; i<graphSize; i++)                 // we check each row 
	{ 
		int nodeColor = graphColors[i]; 
		long numErrorsOnRow = 0; 
		
	
		for (long j=vertexStartList[i]; j<vertexStartList[i+1]; j++){
			if (i == j)  
				continue;  
			
			if (graphColors[compactAdjacencyList[j]] == nodeColor){
				cout << "Color collision from: " << i << " @ " << nodeColor << "  to: " << compactAdjacencyList[j] << " @ " << graphColors[compactAdjacencyList[j]] << endl; 
				numErrors++; 
				numErrorsOnRow++; 
			}

		}


		if (numErrorsOnRow != 0) 
			cout << "Errors for node " << i << " : " << numErrorsOnRow << endl; 
	} 
	
	cout << "Color errors for graph : " << numErrors << endl << "==================== " << endl ;    
} 



//----------------------- The meat -----------------------//
int main(){  
	long maxDegree, numColorsSeq, numColorsParallel, boundaryCount, conflictCount;
	long graphSize, numEdges, subSize;

	graphSize = GRAPHSIZE;
	numEdges = NUMEDGES;
	subSize = graphSize/(GRIDSIZE*BLOCKSIZE);


	long *adjacencyMatrix = new long[graphSize*graphSize*sizeof(long)];  
	long *graphColors = new long[graphSize*sizeof(long)];          
	long *boundaryList = new long[graphSize*sizeof(long)]; 

	long *compactAdjacencyList = new long[(numEdges*2) * sizeof(long)];
	long *vertexStartList = new long[(graphSize+1)*sizeof(long)];



	memset(adjacencyMatrix, 0, graphSize*graphSize*sizeof(long)); 
	memset(graphColors, 0, graphSize*sizeof(long)); 
	memset(boundaryList, 0, graphSize*sizeof(long)); 
	
	conflictCount = boundaryCount = numColorsSeq = numColorsParallel = 0; 
	
	
    long randSeed = time(NULL);
	srand ( randSeed );  // initialize random numbers  
	//srand ( 1271876520 );  // initialize random numbers   
	
	
	cudaEvent_t start, stop, stop_1, stop_4;         
	float elapsedTimeCPU, elapsedTimeGPU, elapsedTimeGPU_1, elapsedTimeGPU_4; 
	
	
	
//--------------------- Graph Creation ---------------------!
	// initialize graph  
	generateMatrix(adjacencyMatrix, numEdges, graphSize);  
	

	// Display graph: Adjacency Matrix
	/*
	cout << "Adjacency Matrix:" << endl; 
	for (long i=0; i<graphSize; i++){  
		for (long j=0; j<graphSize; j++)  
	 		cout << adjacencyMatrix[i*graphSize + j] << "  ";  
	 	cout << endl;  
	}  
	*/ 
	
	// determining the maximum degree  
	maxDegree = getMaxDegree(adjacencyMatrix, graphSize);  
	


	// Get adjacency list
	long *adjacentList = new long[graphSize*maxDegree*sizeof(long)];
	memset(adjacentList, -1, graphSize*maxDegree*sizeof(long)); 

	getAdjacentList(adjacencyMatrix, adjacentList, graphSize, maxDegree);



	// Get Compact adjacency List representation
	getCompactAdjacencyList(adjacentList, compactAdjacencyList, vertexStartList, graphSize, maxDegree);
	vertexStartList[graphSize] = numEdges*2;

	//maxDegree = getBoundaryList(adjacencyMatrix, boundaryList, graphSize, boundaryCount);	// return maxDegree + boundaryCount (as ref param)
	getBoundaryList(compactAdjacencyList, vertexStartList, boundaryList, graphSize, boundaryCount);	// return maxDegree + boundaryCount (as ref param)
	


/*
	cout << "Vertex Start List" << endl;
	for (long j=0; j<graphSize; j++)  
	 	cout << vertexStartList[j] << "  ";  
	 	

	cout << "Compact Adjacency Matrix:" << endl; 
	for (long i=0; i<numEdges*2; i++){  
		cout << compactAdjacencyList[i] << "  ";
	}  


	cout << "Compact Adjacency Matrix 2:" << endl; 
	for (long i=0; i<graphSize; i++){
		cout << endl << i << " : ";
		for (long j=vertexStartList[i]; j<vertexStartList[i+1]; j++){
			cout << compactAdjacencyList[j] << " ";
		}
	}


	for (long i=0; i<graphSize; i++){
		cout << endl <<  i << " : ";
		for (long j=0; j<maxDegree; j++)
			if 	(adjacentList[i*maxDegree +j] != -1)
			{
				cout << adjacentList[i*maxDegree +j] << " ";
			}
			else
				break;	
	}
*/

	//reading the matrix market format sparse matrix
	//getAdjacentListFromSparseMartix_mtx("ch7-8-b4.mtx");



	// Get degree List
	long *degreeList = new long[graphSize*sizeof(long)];
	memset(degreeList, 0, graphSize*sizeof(long)); 

	getDegreeList(adjacentList, degreeList, graphSize, maxDegree);



	
//--------------------- Sequential Graph Coloring ---------------------!
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0);  
	

    //numColorsSeq = colorGraph_FF(adjacentList, graphColors, graphSize, maxDegree);
	colorGraph_FF(compactAdjacencyList, vertexStartList, graphColors, graphSize, maxDegree);  
	//sdoIm(adjacentList, graphColors, degreeList, graphSize, maxDegree);
	//sdoIm(compactAdjacencyList, vertexStartList, graphColors, degreeList, graphSize, maxDegree);
	

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&elapsedTimeCPU, start, stop); 
	

// Get colors
	numColorsSeq = 0;
	for (long i=0; i<graphSize; i++){
		if ( numColorsSeq < graphColors[i] )
			numColorsSeq = graphColors[i];
	}

	

	
//--------------------- Checking for color conflict ---------------------!

	cout << "Sequential Conflict check:";
	//checkCorrectColoring(adjacencyMatrix, graphColors, graphSize); 
	checkCorrectColoring(compactAdjacencyList, vertexStartList, graphColors, graphSize); 
	cout << endl;  
	
	
	
	
//--------------------- Parallel Graph Coloring ---------------------!	
	
	long *conflict = new long[boundaryCount*sizeof(long)];                    // conflict array 
	memset(conflict, 0, boundaryCount*sizeof(long));                        // conflict array initialized to 0  
	
	memset(graphColors, 0, graphSize*sizeof(long));                         // reset colors to 0 
	



//--------------- Steps 1, 2 & 3: Parallel Partitioning + Graph coloring + Conflict Detection
	
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventCreate(&stop_1); 
	cudaEventCreate(&stop_4); 
	cudaEventRecord(start, 0); 
	
		
	long *conflictTmp = new long[boundaryCount*sizeof(long)];
	memset(conflictTmp, 0, boundaryCount*sizeof(long));  
      

	//cudaGraphColoring(adjacentList, boundaryList, graphColors, conflictTmp, boundaryCount, maxDegree);
	cudaGraphColoring(adjacentList, compactAdjacencyList, vertexStartList, boundaryList, graphColors, degreeList, conflictTmp, boundaryCount, maxDegree, graphSize, numEdges);
	
	
	
	cudaEventRecord(stop_1, 0); 
	cudaEventSynchronize(stop_1); 


	long interColorsParallel = 0;
	for (long i=0; i<graphSize; i++){
		if ( interColorsParallel < graphColors[i] )
			interColorsParallel = graphColors[i];
	}




//----- Conflict Count
	for(long i=0; i< boundaryCount; i++)
	{
		long node = conflictTmp[i];
		
		if(node >= 1)
		{
			conflict[conflictCount] = node;
			conflictCount++;
		}
	}
  	delete[] conflictTmp;

   
	cudaEventRecord(stop_4, 0); 
    cudaEventSynchronize(stop_4); 
	



//--------------- Step 4: solve conflicts 
	cout <<"Checkpoint " << endl;
	//conflictSolveFF(adjacentList,  graphSize, conflict, conflictCount, graphColors, maxDegree); 
	//conflictSolveFF(compactAdjacencyList, vertexStartList,  graphSize, conflict, conflictCount, graphColors, maxDegree); 
	
	//conflictSolveSDO(adjacentList, conflict, conflictCount, graphColors,degreeList, graphSize, maxDegree);
	conflictSolveSDO(compactAdjacencyList, vertexStartList, conflict, conflictCount, graphColors,degreeList, graphSize, maxDegree);
	
	
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 

	cudaEventElapsedTime(&elapsedTimeGPU, start, stop); 
	cudaEventElapsedTime(&elapsedTimeGPU_1, start, stop_1); 
	cudaEventElapsedTime(&elapsedTimeGPU_4, start, stop_4); 


	numColorsParallel = 0;
	for (long i=0; i<graphSize; i++){
		if ( numColorsParallel < graphColors[i] )
			numColorsParallel = graphColors[i];
	}


	//conflicts
	/*
	cout << "Conclicts: ";
	for (long i=0; i<conflictCount; i++)
		cout << conflict[i] << " colored " <<  graphColors[conflict[i]] << "    ";
	cout << endl;
	*/


	
	// Display information 
	/*cout << "List of conflicting nodes:"<<endl; 
	 for (long k=0; k<conflictCount; k++)  
	 cout << conflict[k] << "  ";  
	 cout << endl << endl;  */
	

//--------------------- Checking for color conflict ---------------------!

	cout << endl <<  "Parallel Conflict check:";	
	//checkCorrectColoring(adjacencyMatrix, graphColors, graphSize); 
	checkCorrectColoring(compactAdjacencyList, vertexStartList, graphColors, graphSize);	





//--------------------- Parallel Graph Coloring ---------------------!	
	
	
	cout << endl << endl << "Graph Summary" << endl;
	cout << "Vertices: " << graphSize << "   Edges: " << numEdges << "   Density: " << (2*numEdges)/((float)graphSize*(graphSize-1)) << "   Degree: " << maxDegree << endl;
	cout << "Random sed used: " << randSeed << endl;

	cout << endl;
	cout << "Grid Size: " << GRIDSIZE << "    Block Size: " << BLOCKSIZE << "     Total number of threads: " << GRIDSIZE*BLOCKSIZE << endl;
	cout << "Graph Subsize: " << subSize << endl;

	cout << endl;
	cout << "CPU time (Fast Fit): " << elapsedTimeCPU << " ms    -  GPU Time: " << elapsedTimeGPU << " ms" << endl; 
	cout << "ALGO step 1, 2 & 3: " 	<< elapsedTimeGPU_1 << " ms" << endl;  
	cout << "Boundary count: " 		<< elapsedTimeGPU_4 - elapsedTimeGPU_1 << " ms" << endl; 
	cout << "ALGO step 4: " 			<< elapsedTimeGPU - elapsedTimeGPU_4 << " ms" << endl; 

	cout << endl;
	cout << "Boundary Count: " << boundaryCount << endl;
	cout << "Conflict count: " << conflictCount << endl;

	cout << endl;
    cout << "Colors before solving conflict: " << interColorsParallel << endl;
	cout << "Sequential Colors: " << numColorsSeq << "      -       Parallel Colors: " << numColorsParallel << endl;     
	
	


	
	
//--------------------- Cleanup ---------------------!		

	delete[] adjacencyMatrix; 
	delete[] graphColors; 
	delete[] conflict; 
	delete[] boundaryList;
	delete[] adjacentList;
	
	return 0;  
}  

