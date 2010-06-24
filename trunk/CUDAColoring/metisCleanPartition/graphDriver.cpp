// Graph coloring 
#include <ctime> 
#include <stdio.h>  
#include <stdlib.h>  
#include <time.h>  
#include <iostream> 
#include <math.h>  
#include <set> 
#include <assert.h>
#include <fstream>

#include "graphColoring.h" 

using namespace std;  


//----------------------- Utilities -----------------------//
int findPower(int x){
	int num = 2;
	int powerIndex = 1;
	
	while (num <= x){
		powerIndex++;
		num = pow(2,powerIndex);
	}
	
	cout << "Closest power: " << num << endl;
	return num;
}


int findMultiple(int multipleOf, int x){
	int base = multipleOf;
	int powerIndex = 0;
	int num = 0;
	
	while (num <= x){
		powerIndex++;
		num = base * powerIndex;	
	}
	
	cout << "Closest multiple of " << base << ": " << num << endl;
	return num;
}


int inline min(int n1, int n2) 
{ 
	if (n1>=n2) 
		return n2; 
	else 
		return n1; 
} 




//----------------------- Graph initializations -----------------------//
void readGraph(int *&adjacencyMatrix, const char *filename, int _gridSize, int _blockSize, int &graphSizeRead, int &graphSize, long &edgeSize){
	char comments[512];
	int graphSizeX, graphSizeY, from, to, numEdges, weightedGraph;
	float weight;
	
	
	
	numEdges = 0;
	
	ifstream graphFile(filename);
	
	
	if (graphFile.is_open())
	{
		while (graphFile.peek() == '%'){
			graphFile.getline(comments,512);
		}
		
		graphFile >> graphSizeX >> graphSizeY >> edgeSize;
		cout << "Rows: " << graphSizeX << "  ,  Columns: " <<  graphSizeY << "   - Number of edges: " << edgeSize << endl;
		
		if (graphSizeX != graphSizeY){
			cout << "Non Symmetric graph!" << endl;
			exit(1);
		}
		else 
		{
			cout << "Is it a weighted graph(1:yes  -  0:no): ";
			cin >> weightedGraph;
	
			
			graphSizeRead = graphSizeX;
			graphSize = findMultiple(_gridSize*_blockSize, graphSizeRead);
			
			adjacencyMatrix = new int[graphSize * graphSize];
			memset(adjacencyMatrix, 0, graphSize * graphSize *sizeof(int));
			
			for (int i=0; i<edgeSize; i++){
				if (weightedGraph == 1)
					graphFile >> from >> to >> weight;	
				else
					graphFile >> from >> to;
				
				if (!(from == to)){
					numEdges++;
					adjacencyMatrix[(from-1)*graphSize + (to-1)] = 1;
					adjacencyMatrix[(to-1)*graphSize + (from-1)] = 1;
				
					/*
					if (weightedGraph == 1)
						cout << from << " , " << to << " : " << weight << endl; 
					else
						cout << from << " , " << to << endl;
					*/
				}
			}
		}
	}
	else {
		cout << "Reading " << filename << " failed!" << endl;
		exit(1);
	}
	
	edgeSize = numEdges;
	cout << "Graph: " << graphSizeRead << " - " <<  graphSize << " - " <<  edgeSize << endl;
	cout << "File " << filename << " was successfully read!" << endl;
}





//----------------------- Display -----------------------//
// Author: Pascal
// Displays an adjacencyList
void displayAdjacencyList(int *adjacencyList, int graphSize, int maxDegree){
	cout << endl << "Adjacency List:" << endl;
	for (int i=0; i<graphSize; i++){
		cout << i << ": ";
		
		for (int j=0; j<maxDegree; j++){
			if (adjacencyList[i*maxDegree + j] != -1)
				cout << adjacencyList[i*maxDegree + j] << " ";
			else 
				break;
		}
		
		cout << endl;
	}
}










//----------------------- Graph initializations -----------------------//

// Author: Pascal 
// Genetates a graph 
void generateMatrix(int *matrix, int graphSize, int num){  
	int x, y;  
	int count = 0;  
	
	while (count < num){  
		x = rand()%graphSize;  
		y = rand()%graphSize;  
		
		if (matrix[x*graphSize + y] != 1)          // if not already assigned an edge 
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
void getAdjacentList(int *adjacencyMatrix, int *adjacentList, int size, int maxDegree)
{
	for (int i=0; i<size; i++){ 
		int nbCount = 0;
		for (int j=0; j<size; j++){
			if ( adjacencyMatrix[i*size + j] == 1)  
			{
				adjacentList[i*maxDegree + nbCount] = j;
				nbCount++;
			}
		}
	}
}




// Author: Pascal 
// get the degree information for a graph 
int getMaxDegree(int *adjacencyMatrix, int size){  
	cout << "size : " <<  size << endl;
	int maxDegree = 0;   
	int degree;  
	
	for (int i=0; i<size; i++){  
		degree = 0;  
		
		for (int j=0; j<size; j++){         
			if (adjacencyMatrix[i*size + j] == 1){
				degree++; 
			//	cout << i << " , " << j << endl;
			}
			
		}
		
		if (degree > maxDegree)  
			maxDegree = degree;  
	}  
	
	return maxDegree;  
}  




// Author: Pascal
// get the degree of each element in the graph and returns the maximum degree
void getDegreeList(int *adjacencyList, int *degreeList, int sizeGraph, int maxDegree){
    for (int i=0; i<sizeGraph; i++){
        int count = 0;
        
        for (int j=0; j<maxDegree; j++){
            if (adjacencyList[i*maxDegree + j] != -1)
                count++;
            else
                break;  
        }
		
        degreeList[i] = count;
    }
}



// Author: Peihong
int getBoundaryList(int *adjacencyMatrix, int *boundaryList, int size, int &boundaryCount, int graphSize, int _gridSize, int _blockSize){  
	int maxDegree = 0;   
	int degree;  
	
	set<int> boundarySet; 
	boundarySet.clear(); 
	
	int subSize = graphSize/(_gridSize*_blockSize);
	cout << "SubSize =" << subSize << endl;
	
	for (int i=0; i<size; i++){  
		degree = 0;  
		
		int subIdx = i/(float)subSize;
		int start = subIdx * subSize;
		int end = min( (subIdx + 1)*subSize, size );
		
		for (int j=0; j<size; j++){           
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
	
	set<int>::iterator it = boundarySet.begin(); 
	for (int i=0; it != boundarySet.end(); it++)  
	{ 
		boundaryList[i] = *it;
		i++; 
	}  
	
	return maxDegree;  
}  



int getBoundaryList(int *adjacencyList, int *boundaryList, int graphSize, int maxDegree, int _gridSize, int _blockSize, int *startPartitionList, int *endPartitionList){  
	int boundaryCount = 0;
	set<int> boundarySet; 
	boundarySet.clear(); 
	
	int start, end;
	int partitionIndex = 0; 
	//int subSize = graphSize/(_gridSize*_blockSize);
	//cout << "SubSize = " << subSize << endl;
	
	for (int i=0; i<graphSize; i++)
	{  
		//int subIdx = i/(float)subSize;
		//start = subIdx * subSize;
		//end = min( (subIdx + 1)*subSize, subSize );
		
		if (!(i < endPartitionList[partitionIndex]))
			partitionIndex++;
		
		start = startPartitionList[partitionIndex];
		end = endPartitionList[partitionIndex];
		
		
		
		for (int j=0; j<maxDegree; j++)           
			if (adjacencyList[i*maxDegree + j] != -1) 
				if ((j < start) || (j >= end))
					boundarySet.insert(i);
		
	} 
	
	boundaryCount = boundarySet.size();
	
	set<int>::iterator it = boundarySet.begin(); 
	for (int i=0; it != boundarySet.end(); it++)  
	{ 
		boundaryList[i] = *it;
		i++; 
	}  
	
	return boundaryCount;  
} 




//----------------------- Fast Fit Graph Coloring -----------------------//

// Author: Pascal & Shusen
// GraphColor Adjacency list
int colorGraph_FF(int *list, int *colors, int size, int maxDegree){  
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
int degree(int vertex, int *degreeList){
	return degreeList[vertex];
}



// Author: Pascal
// return the saturation of that node
int saturation(int vertex, int *adjacencyList, int *graphColors, int maxDegree){
    int saturation = 0;
    int *colors = new int[maxDegree+1];
	
    memset(colors, 0, (maxDegree+1)*sizeof(int));           // initialize array
	
	
    for (int i=0; i<maxDegree; i++){
        if (adjacencyList[vertex*maxDegree + i] != -1)
			//  colors[ graphColors[vertex] ] = 1;                      // at each colored set the array to 1
			colors[ graphColors[adjacencyList[vertex*maxDegree + i]] ] = 1;                      // at each colored set the array to 1
        else
            break;
    }
	
	
    for (int i=1; i<maxDegree+1; i++)                                       // count the number of 1's but skip uncolored
        if (colors[i] == 1)
            saturation++;
	
	delete[] colors; 	
	
    return saturation;
}




// Author: Pascal
// colors the vertex with the min possible color
int color(int vertex, int *adjacencyList, int *graphColors, int maxDegree, int numColored){
    int *colors = new int[maxDegree + 1];
    memset(colors, 0, (maxDegree+1)*sizeof(int));   
    
    if (graphColors[vertex] == 0)
        numColored++;
	//	else
	//		cout << "Old color: " << graphColors[vertex] << "   ";
    
    for (int i=0; i<maxDegree; i++)                                         // set the index of the color to 1
        if (adjacencyList[vertex*maxDegree + i] != -1)
            colors[  graphColors[  adjacencyList[vertex*maxDegree + i]  ]  ] = 1;
        else {
            break;
        }
	
    
	
    for (int i=1; i<maxDegree+1; i++)                                       // nodes still equal to 0 are unassigned
        if (colors[i] != 1){
            graphColors[vertex] = i;
			//			cout << " New color:" << i << endl;
            break;
        }
    
	delete[] colors; 
	
    return numColored;
}




// Author: Pascal
// main driver function for graph coloring
int sdoIm(int *adjacencyList, int *graphColors, int *degreeList, int sizeGraph, int maxDegree){
    int satDegree, numColored, max, index;
    numColored = 0;
    int iterations = 0;
    
	
    while (numColored < sizeGraph){
        max = -1;
        
        for (int i=0; i<sizeGraph; i++){
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
				
				numColored = color(index,adjacencyList,graphColors, maxDegree, numColored);
				iterations++;
            } 
        }
    }
    
    return iterations;
}






//----------------------- Conflict Solve -----------------------//

// Author: Pascal
void conflictSolveSDO(int *adjacencyList, int *conflict, int conflictSize, int *graphColors, int *degreeList, int sizeGraph, int maxDegree){
    int satDegree, numColored, max, index;
    numColored = 0;
	
	// Set their color to 0
	for (int i=0; i<conflictSize; i++)
		graphColors[conflict[i]-1] = 0;
    
	
    while (numColored < conflictSize){
        max = -1;
        
        for (int i=0; i<conflictSize; i++){
			int vertex = conflict[i]-1;
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



// Author: Pascal & Shusen
// Solves conflicts using Fast Fit
void conflictSolveFF(int *Adjlist, int size, int *conflict, int conflictSize, int *graphColors, int maxDegree){
	int i, j, vertex, *colorList, *setColors;
	colorList = new int[maxDegree];
	setColors = new int[maxDegree];
	
	
	// assign colors up to maxDegree in setColors
	for (i=0; i<maxDegree; i++){
		setColors[i] = i+1;
	}
	
	
	for (i=0; i<conflictSize; i++){
        memcpy(colorList, setColors, maxDegree*sizeof(int));                    // set the colors in colorList to be same as setColors
		
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





//----------------------- Checking for error -----------------------//
// Checking if coloring has been done properly...

// Author: Pascal 
// Checks if a graph has conflicts or not from Adjacency Matrix
void checkCorrectColoring(int *adjacencyMatrix, int *graphColors, int graphSize){ 
	int numErrors = 0; 
	
	cout << endl << "==================" << endl << "Error checking for Graph" << endl; 
	
	for (int i=0; i<graphSize; i++)                 // we check each row 
	{ 
		int nodeColor = graphColors[i]; 
		int numErrorsOnRow = 0; 
		
		for (int j=0; j<graphSize;j++){ // check each column in the matrix 
			
			// skip itself 
			if (i == j) 
				continue; 
			
			if (adjacencyMatrix[i*graphSize + j] == 1)      // there is a connection to that node 
				if (graphColors[j] == nodeColor) 
				{ 
					cout << "Color collision from node: " << i << " colored with: " << nodeColor << "  to node: " << j << " colored with " << graphColors[j] << endl; 
					numErrors++; 
					numErrorsOnRow++; 
				} 
		} 
		
		if (numErrorsOnRow != 0) 
			cout << "Errors for node " << i << " : " << numErrorsOnRow << endl; 
	} 
	
	cout << "Color errors for graph : " << numErrors << endl << "==================== " << endl ;    
} 



// Author: Pascal 
// Checks if a graph has conflicts or not from adjacency List
void checkCorrectColoring(int *adjacencyList, int *graphColors, int graphSize, int maxDegree){
    int numErrors = 0;
	
    cout << endl << "==================" << endl << "Error checking for Graph" << endl;
	
    for (int i=0; i<graphSize; i++)                 // we check each row
    {
        int nodeColor = graphColors[i];
        int numErrorsOnRow = 0;
		
        for (int j=0; j<maxDegree;j++){
			
            if (adjacencyList[i*maxDegree + j] == -1)
                break;
            else{     // there is a connection to that node
                int node = adjacencyList[i*maxDegree + j];
                if (graphColors[node] == nodeColor)
                {
                    cout << "Color collision from node: " << i << " col with " << nodeColor << "    to: " << node << " col with " << graphColors[node] << endl;
                    numErrors++;
                    numErrorsOnRow++;
                }
            }
        }
		
        if (numErrorsOnRow != 0)
            cout << "Errors for node " << i << " : " << numErrorsOnRow << endl;
    }
	
    cout << "Color errors for graph : " << numErrors << endl << "==================== " << endl ;   
}





//----------------------- Metis -----------------------//
// Metis related stuff...

// Author: Pascal
// Creates outpuf for metis file
void createMetisInput(int *adjacencyList, int graphSize, int numEdges, int maxDegree){
	string metisInputFilename;
	
	cout << endl << "Creating file to send to metis ..." << endl;
	cout << "Enter filename for file: ";
	cin >> metisInputFilename;
	
	cout << "Graph Size: " << graphSize << endl;
	cout << "Num Edges: " << numEdges << endl;
	

	ofstream myfile (metisInputFilename.c_str());
  	if (myfile.is_open())
  	{
		myfile << graphSize << " " << numEdges << "\n";
		
		for (int i=0; i<graphSize; i++){ 
			
			for (int j=0; j<maxDegree; j++)  
				if (adjacencyList[i*maxDegree + j] == -1)
					break;
				else
					myfile << (adjacencyList[i*maxDegree + j]) + 1 << " ";
			
			myfile << endl;  
		}  
		
		myfile.close();
  	}
  	else {
		cout << "Unable to open file to write";
		exit(0);
	}
}
	
	
	
	
// Author: Pascal & Peihong 
// Reads in metis partitioned file
void readMetisOutput(int *partitionList, int graphSize){
	string metisOutputFilename;

	
	cout << endl << "Reading partitioned metis file..." << endl;
	cout << "Enter filename to read from (e.g. metisInput2048.txt.part.256): ";
	cin >> metisOutputFilename;
	
	
	ifstream metisFile(metisOutputFilename.c_str());
	if (metisFile.is_open()){
		for(int i=0; i<graphSize; i++)
		{
			metisFile >> partitionList[i];
		}
		metisFile.close();
	
		/*
		cout << "Partition List" << endl;
		for(int i=0; i<graphSize; i++)
			cout << i << ":: " << partitionList[i] << endl;
		 */
		
	}
	else {
		cout << "Reading in file failed" << endl;
		exit(0);
	}
}



// Author: Pascal
// Output file for use by metis
// Input partitioned file
// gets the new adjacency list
int metis(int *adjacencyList, int *newAdjacencyList, int graphSize, int numEdges, int maxDegree, int *startPartitionList, int *endPartitionList){
	int *partitionList = new int[graphSize];
	int *newGraphOrdering = new int[graphSize];
	
	int *adjacencyListOrg = new int[graphSize*maxDegree];	// created so as not to modify the original adjacency List
	memcpy(adjacencyListOrg, adjacencyList, graphSize*maxDegree*sizeof(int));
	
	
	int currentPosition = 0;
	int numPartitions = 256;
	
	memset(newAdjacencyList, -1, graphSize*maxDegree*sizeof(int));
	
	
	
	createMetisInput(adjacencyListOrg, graphSize, numEdges, maxDegree);
	
	cout << endl << "Enter the number of partitions used in metis: ";
	cin >> numPartitions;
	
		
	readMetisOutput(partitionList, graphSize);


	
	// Get the maximum and minimum in each partition
	int min, max, count, partitionMin, partitionMax, startPartitionCount ,endPartitionCount;
	min = 1000000;
	max = -1;
	partitionMin = partitionMax = -1;
	startPartitionCount = endPartitionCount = 0;
	
	for (int i=0; i<numPartitions; i++){
		count = 0;
		startPartitionCount = endPartitionCount;
		
		for (int j=0; j<graphSize; j++){
			if (partitionList[j] == i){
				count++;
			}
		}
		endPartitionCount += count;
		 
		startPartitionList[i] = startPartitionCount;
		endPartitionList[i] = endPartitionCount;
		
		if (count > max){
			max = count;
			partitionMax = i;
		}
		
		if (count < min){
			min = count;
			partitionMin = i;
		}
	}
		
	cout << "Min in partiton: " << min << "  for partition: " << partitionMin << endl;
	cout << "Max in parition: " << max << "  for partition: " << partitionMax << endl;
	
	cout << "Partitions list:" << endl;
	for (int i=0; i<numPartitions; i++)
		cout << i << "-   start: " << startPartitionList[i] << "    end: " << endPartitionList[i] 
					<<  "  size: " << (endPartitionList[i] - startPartitionList[i]) << endl;
	cout << endl;
	
	
	
	// Gets the new Ordering of the nodes
	for (int i=0; i<numPartitions; i++)
		for (int j=0; j<graphSize; j++){
			if (partitionList[j] == i){
				newGraphOrdering[j] = currentPosition;
				currentPosition++;
			}
		}
	
	/*
	cout << "New Ordering" << endl;
	for (int i=0; i<graphSize; i++)
	{
		cout << i << ": " << newGraphOrdering[i] << endl;
	}
	*/
	
	
	
	// Replaces the nodes in the adjacency list by the new ordering numbers
	for (int i=0; i<graphSize; i++){
		for (int j=0; j<maxDegree; j++){
			int node = adjacencyListOrg[i*maxDegree + j];
			
			if (adjacencyListOrg[i*maxDegree + j] != -1){
				adjacencyListOrg[i*maxDegree + j] = newGraphOrdering[node];
			}
			else
				break;
		}	
	}
	
	
	
	// Places the different lists of the adjacency list in the right place
	for (int i=0; i<graphSize; i++){
		int newPosn = newGraphOrdering[i];
		
		for (int j=0; j<maxDegree; j++){
			if 	(adjacencyListOrg[i*maxDegree + j] != -1)
				newAdjacencyList[newPosn*maxDegree + j] = adjacencyListOrg[i*maxDegree + j];
			else
				break;
		}	
	}
	
	/*
	cout << "New Adjacency List:" << endl; 
	displayAdjacencyList(newAdjacencyList, graphSize, maxDegree);
	*/
	
	delete []adjacencyListOrg;
	delete []partitionList;
	delete []newGraphOrdering;

	return numPartitions;
}




//----------------------- Other -----------------------//
// Any additional stuff needed ....





//----------------------- The meat -----------------------//

int main(int argc, char *argv[]){  
	if (argc != 4){
		cout << "Arguments passed: " << argc << endl;
		cout << "3 Arguments needed: " << endl 
			<< "cuExe <passes> <atificial (0 => input file)>  <metis (1 => use metis)>" << endl 
			<< "e.g. cuExe 1 0 1" << endl;

		return 1;
	}
	
	int maxDegree, numColorsSeq, numColorsParallel, boundaryCount, conflictCount, passes, graphSize, graphSizeRead;
	int _gridSize, _blockSize, numMetisPartitions;
	float density;
	long numEdges;
	string inputFilename;
	
	
	int *adjacentList, *adjacencyMatrix;
	
	conflictCount = boundaryCount = numColorsSeq = numColorsParallel = 0; 
	
	
	
	//--------------------- Parameter initialization ---------------------!
	bool useMetis = true;
	bool sdo = true;
	bool sdoConflictSolver = true;
	
	
	long randSeed = time(NULL);	
	randSeed = 1272167817;			// to set to a specific random seed for replicability
	

	passes = atoi(argv[1]);		// get number of passes
	
	if (atoi(argv[3]) == 1)
		useMetis = true;
	else
		useMetis = false;
	

	
	bool artificial = false;
	if (atoi(argv[2]) == 0)
		artificial = false;
	else
		artificial = true;
	
	
		
	//--------------------- Graph Creation ---------------------!

	// Grid and block size
	cout << endl << "!--------------- Graph Coloring program -------------------!" << endl;
	cout << "Enter grid size (e.g 4): ";
	cin >> _gridSize;
	cout << "Enter block size (e.g 64): ";
	cin >> _blockSize;
	cout << endl;
	
	int *startPartitionList = new int[_gridSize*_blockSize];	
	int *endPartitionList = new int[_gridSize*_blockSize];		
	memset(startPartitionList, -1, _gridSize*_blockSize*sizeof(int));
	memset(endPartitionList, -1, _gridSize*_blockSize*sizeof(int));
	
	int numRandoms = _gridSize*_blockSize*10;

	
	// Artificial or real 
	if (artificial == false){	// input file required - returns an adjacency list of the graph, graph size and max degree
		ifstream testFile;	
	
		cout << "Enter graph input filename (e.g. 1138_bus.mtx): ";
		cin >> inputFilename;
		
		// gets a compact adjacency list from the file input
		readGraph(adjacencyMatrix, inputFilename.c_str(), _gridSize, _blockSize, graphSizeRead, graphSize, numEdges);
		cout << graphSizeRead << " - " << graphSize << " - " << numEdges << endl; 
		
		
		// gets the max degree
		maxDegree = getMaxDegree(adjacencyMatrix, graphSize);
		cout << "Got degree: " << maxDegree << endl;
		
		
		// Get adjacency list
		adjacentList = new int[graphSize*maxDegree];
		memset(adjacentList, -1, graphSize*maxDegree*sizeof(int)); 
	
		getAdjacentList(adjacencyMatrix, adjacentList, graphSize, maxDegree);
	}
	else
	{
		cout << "Enter graph size: ";
		cin >> graphSize;
		
		cout << "Enter density: ";
		cin >> density;
		
		numEdges = density*graphSize*(graphSize-1)/2;
		
		
		adjacencyMatrix = new int[graphSize*graphSize];  
		memset(adjacencyMatrix, 0, graphSize*graphSize*sizeof(int)); 
		
		
		srand ( randSeed );  // initialize random numbers  
		  
		
		// generates a graph
		generateMatrix(adjacencyMatrix, graphSize, numEdges);
		
		
		// gets the max degree
		maxDegree = getMaxDegree(adjacencyMatrix, graphSize);
		
		
		
		// Get adjacency list
		adjacentList = new int[graphSize*maxDegree];
		memset(adjacentList, -1, graphSize*maxDegree*sizeof(int)); 
		
		getAdjacentList(adjacencyMatrix, adjacentList, graphSize, maxDegree);
	}
	
	cout << "Allocation successful!" << endl;
	
	delete []adjacencyMatrix;
	adjacencyMatrix = NULL;
	
	
	// Some further intializations
	int *graphColors = new int[graphSize];          
	int *boundaryList = new int[graphSize]; 
	int *degreeList = new int[graphSize];
	
	memset(graphColors, 0, graphSize*sizeof(int)); 
	memset(boundaryList, 0, graphSize*sizeof(int)); 
	memset(degreeList, 0, graphSize*sizeof(int)); 
	
	
	// Get degree List
	getDegreeList(adjacentList, degreeList, graphSize, maxDegree);
	
	
	
	int *randomList = new int[numRandoms];
	for (int i=0; i<numRandoms; i++)		// stores random numbers in the range of 0 to 2
		randomList[i] = rand()%2;
	
	
	
	//--------------------- Metis ---------------------!
	
	int *metisAdjacencyList = new int[graphSize*maxDegree];
	int *metisDegreeList = new int[graphSize];
	
	
	
	if (useMetis == true){
		memset(metisAdjacencyList, 0, graphSize*maxDegree*sizeof(int)); 
		memset(metisDegreeList, 0, graphSize*sizeof(int)); 
			
		numMetisPartitions = metis(adjacentList, metisAdjacencyList, graphSize, numEdges, maxDegree, startPartitionList, endPartitionList);	// Metis
	
		getDegreeList(metisAdjacencyList, metisDegreeList, graphSize, maxDegree);
		
		memcpy(adjacentList, metisAdjacencyList, graphSize*maxDegree*sizeof(int));
		memcpy(degreeList, metisDegreeList, graphSize*sizeof(int));
	}else{
		// allocating exact partition size for
		int partitionSizes = graphSize / (_gridSize*_blockSize);
		
		for (int i=0; i<_gridSize*_blockSize; i++){
			startPartitionList[i] = i*partitionSizes; 
			endPartitionList[i] = (i+1)*partitionSizes;
		}
	}
	
	
	
	
	//--------------------- Boundary List ---------------------!
	///////// Needs to be updated to use adjacencyList instead of adjacencyMatrix!!!!!!!!!!!!!!!!!!!
	cudaEvent_t start_b, stop_b;
	float elapsedTimeBoundary;
	cudaEventCreate(&start_b); 
	cudaEventCreate(&stop_b); 
	cudaEventRecord(start_b, 0); 
	
	//maxDegree = getBoundaryList(adjacencyMatrix, boundaryList, graphSize, boundaryCount, graphSize, _gridSize, _blockSize);	// return maxDegree + boundaryCount (as ref param)
	boundaryCount = getBoundaryList(adjacentList, boundaryList, graphSize, maxDegree, _gridSize, 
									_blockSize, startPartitionList, endPartitionList);				// get boundaryCount and get boundary list
	
	
	cudaEventRecord(stop_b, 0); 
	cudaEventSynchronize(stop_b); 
	cudaEventElapsedTime(&elapsedTimeBoundary, start_b, stop_b); 
	cout << "Get boundaryList :"<< elapsedTimeBoundary << " ms" << endl;

	

	
	
	
	//--------------------- Sequential Graph Coloring ---------------------!
	cudaEvent_t start, stop, stop_1, stop_4;         
	float elapsedTimeCPU, elapsedTimeGPU, elapsedTimeGPU_1, elapsedTimeGPU_4; 
	
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0);  
	
	
	
	// Original adjacency List
	if (sdo == true)
		sdoIm(adjacentList, graphColors, degreeList, graphSize, maxDegree);
	else
		colorGraph_FF(adjacentList, graphColors, graphSize, maxDegree);  
	
	
	
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&elapsedTimeCPU, start, stop); 
	
	
	// Get colors
	numColorsSeq = 0;
	for (int i=0; i<graphSize; i++){
		if ( numColorsSeq < graphColors[i] )
			numColorsSeq = graphColors[i];
	}
	
	
	

	//--------------------- Checking for color conflict ---------------------!
	
	cout << endl << "Sequential Conflict check: ";
	checkCorrectColoring(adjacentList, graphColors, graphSize, maxDegree);
	
	cout << endl;  
	cout << "Sequential colors: " << numColorsSeq << endl;

	
	
	
	
	
	//--------------------- Parallel Graph Coloring ---------------------!	
	
	int *conflict = new int[boundaryCount];                    // conflict array 
	memset(conflict, 0, boundaryCount*sizeof(int));                        // conflict array initialized to 0  
	
	memset(graphColors, 0, graphSize*sizeof(int));                         // reset colors to 0 
	
	
	
	
	//--------------- Steps 1, 2 & 3: Parallel Partitioning + Graph coloring + Conflict Detection
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventCreate(&stop_1); 
	cudaEventCreate(&stop_4); 
	cudaEventRecord(start, 0); 
	
	
	int *conflictTmp = new int[boundaryCount];
	memset(conflictTmp, 0, boundaryCount*sizeof(int));  
	
	cudaGraphColoring(adjacentList, boundaryList, graphColors, degreeList, conflictTmp, boundaryCount, 
					  maxDegree, graphSize, passes, _gridSize*_blockSize, _gridSize, _blockSize,
					  startPartitionList, endPartitionList, randomList, numRandoms);
	
	
	cudaEventRecord(stop_1, 0); 
	cudaEventSynchronize(stop_1); 
	
	
	
	// count number of parallel colors
	int interColorsParallel = 0;
	for (int i=0; i<graphSize; i++)
		if ( interColorsParallel < graphColors[i] )
			interColorsParallel = graphColors[i];
	
	
	
	
	
	//-------- Conflict Count
	for (int i=0; i< boundaryCount; i++)
	{
		int node = conflictTmp[i];
		
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
	//cout <<"Checkpoint " << endl;
	
	
	if (sdoConflictSolver == true)
		conflictSolveSDO(adjacentList, conflict, conflictCount, graphColors,degreeList, graphSize, maxDegree);
	else
		conflictSolveFF(adjacentList,  graphSize, conflict, conflictCount, graphColors, maxDegree); 
	
	
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop); 
	
	
	cudaEventElapsedTime(&elapsedTimeGPU, start, stop); 
	cudaEventElapsedTime(&elapsedTimeGPU_1, start, stop_1); 
	cudaEventElapsedTime(&elapsedTimeGPU_4, start, stop_4); 
	
	
	numColorsParallel = 0;
	for (int i=0; i<graphSize; i++){
		if ( numColorsParallel < graphColors[i] )
			numColorsParallel = graphColors[i];
	}
	
	
	//conflicts
	/*
	 cout << "Conclicts: ";
	 for (int i=0; i<conflictCount; i++)
		cout << conflict[i] << " colored " <<  graphColors[conflict[i]] << "    ";
	 cout << endl;
	 */
	
	
	
	// Display information 
	/*
	 cout << "List of conflicting nodes:"<<endl; 
	 for (int k=0; k<conflictCount; k++)  
		cout << conflict[k] << "  ";  
	 cout << endl << endl;  
	 */
	
	
	
	//--------------------- Checking for color conflict ---------------------!
	
	cout << endl <<  "Parallel Conflict check:";	
	
	//checkCorrectColoring(adjacencyMatrix, graphColors, graphSize); 	
	checkCorrectColoring(adjacentList, graphColors, graphSize, maxDegree);

	
	
	
	
	
	
	
	//--------------------- Information Output ---------------------!	
	
	
	cout << endl << endl << "!------- Graph Summary:" << endl;
	cout << "Vertices: " << graphSize << "   Edges: " << numEdges << "   Density: " << (2*numEdges)/((float)graphSize*(graphSize-1)) << "   Degree: " << maxDegree << endl;
	if (artificial == false){
		cout << "Graph read in: " << inputFilename << endl;
		cout << "Vertices in graph: " << graphSizeRead << endl;
	}
	else
		cout << "Random seed used: " << randSeed << endl;
	cout << endl;
	
	
	cout << "Grid Size: " << _gridSize << "    Block Size: " << _blockSize << "     Total number of threads: " << _gridSize*_blockSize << endl;
	cout << "Graph average subsize: " << graphSize/(_gridSize*_blockSize) << endl;
	
	if (useMetis == true)
		cout << "Number of metis partitions: " << numMetisPartitions << endl;
	cout << endl;
	
	cout << "GPU Passes done: " << passes << endl;
	
	if (sdo == true)
		if (sdoConflictSolver == true)
			cout << "CPU time (SDO): " << elapsedTimeCPU << " ms    -  GPU Time (SDO Solver): " << elapsedTimeGPU << " ms" << endl; 
		else
			cout << "CPU time (SDO): " << elapsedTimeCPU << " ms    -  GPU Time (FF Solver): " << elapsedTimeGPU << " ms" << endl; 
        else
			if (sdoConflictSolver == true)
				cout << "CPU time (First Fit): " << elapsedTimeCPU << " ms    -  GPU Time (SDO Solver): " << elapsedTimeGPU << " ms" << endl; 
			else
				cout << "CPU time (First Fit): " << elapsedTimeCPU << " ms    -  GPU Time (FF Solver): " << elapsedTimeGPU << " ms" << endl; 
	
	
	cout << endl << "Getting boundary list: " 	<< elapsedTimeBoundary << " ms" << endl; 
	cout << "ALGO step 1, 2 & 3   : " 	<< elapsedTimeGPU_1 << " ms" << endl;  
	cout << "Boundary count       : " 	<< elapsedTimeGPU_4 - elapsedTimeGPU_1 << " ms" << endl; 
	cout << "ALGO step 4          : " 	<< elapsedTimeGPU   - elapsedTimeGPU_4 << " ms" << endl; 
	cout << "Total time           : "	<< (elapsedTimeBoundary + elapsedTimeGPU) << " ms" << endl;
	cout << endl;
	
	
	cout << "Boundary Count: " << boundaryCount << endl;
	cout << "Conflict count: " << conflictCount << endl;
	cout << endl;
	
	
	cout << "Colors before solving conflict: " << interColorsParallel << endl;
	cout << "Sequential Colors: " << numColorsSeq << "      -       Parallel Colors: " << numColorsParallel << endl;     
	cout <<"GPU speed up (including boundary): "<< (elapsedTimeBoundary + elapsedTimeGPU)/elapsedTimeGPU << " x" << endl;

	cout << "||=============================================================||" << endl << endl;

	
	//--------------------- Cleanup ---------------------!		
	
	 
	delete []graphColors; 
	delete []conflict; 
	delete []boundaryList;
	delete []adjacentList;
	delete []degreeList;
	delete []randomList;

	return 0;  
}  

