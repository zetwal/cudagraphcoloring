// g++ sdo.cpp -o sdo

#include <iostream>
using namespace std;

const int GRAPHSIZE = 1000;
const int NUMEDGES = 400000;

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
		
		if (matrix[x*size + y] != 1)            // if not already assigned an edge 
			if (x != y){  
				matrix[x*size + y] = 1;       // non directional graph  
				matrix[y*size + x] = 1;  
				count++;  
			}  
	}  
}  


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


//Author:Peihong
// node index start from 1
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
		//cout<<"number of neighbor count="<<nbCount<<endl;
	}
}



// get the degree of each element in the graph
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


// returns the degree of that node
int degree(int vertex, int *degreeList){
	return degreeList[vertex];
}




// return the saturation of that node
int saturation(int vertex, int *adjacencyList, int *graphColors, int maxDegree){
	int saturation = 0;
	int *colors = new int[maxDegree+1];

	memset(colors, 0, (maxDegree+1)*sizeof(int));		// initialize array


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
int color(int vertex, int *adjacencyList, int *graphColors, int maxDegree, int numColored){
	int *colors = new int[maxDegree + 1];
	memset(colors, 0, (maxDegree+1)*sizeof(int));	
	
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


int sdoIm(int *adjacencyList, int *graphColors, int *degreeList, int sizeGraph, int maxDegree){
	int satDegree, numColored, max, index;
	numColored = 0;
	int iterations = 0;
	

	while (numColored < sizeGraph){
		max = -1;
		
		for (int i=0; i<sizeGraph; i++){
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
			iterations++;
		}
	}
	
	return iterations;
}



void checkCorrectColoring(int *adjacencyMatrix, int *graphColors){ 
	int numErrors = 0; 
	
	cout << endl << "==================" << endl << "Error checking for Graph" << endl; 
	
	for (int i=0; i<GRAPHSIZE; i++)                 // we check each row 
	{ 
		int nodeColor = graphColors[i]; 
		int numErrorsOnRow = 0; 
		
		for (int j=0; j<GRAPHSIZE;j++){ // check each column in the matrix 
			
			// skip itself 
			if (i == j) 
				continue; 
			
			if (adjacencyMatrix[i*GRAPHSIZE + j] == 1)      // there is a connection to that node 
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
	
	cout << "Color errors for graph : " << numErrors << endl << "====================" << endl ;    
}



// First Fit - simplest one ever 
int firstFit(int *matrix, int *colors, int size, int maxDegree){ 
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
			if (matrix[i*size + j] == 1) 
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



int main(){
	int maxDegree, numColors, numColors2;
	int *adjacencyMatrix = new int[GRAPHSIZE*GRAPHSIZE*sizeof(int)];  
	memset(adjacencyMatrix, 0, GRAPHSIZE*GRAPHSIZE*sizeof(int));

	generateMatrix(adjacencyMatrix, GRAPHSIZE, NUMEDGES); 
	maxDegree = getMaxDegree(adjacencyMatrix, GRAPHSIZE);

	

	int *adjacencyList = new int[GRAPHSIZE*maxDegree*sizeof(int)];  
	int *graphColors = new int[GRAPHSIZE*sizeof(int)]; 
	int *graphColors2 = new int[GRAPHSIZE*sizeof(int)]; 
	int *boundaryList = new int[GRAPHSIZE*sizeof(int)]; 
	int *degreeList = new int[GRAPHSIZE*sizeof(int)]; 

	 
	memset(adjacencyList, -1, GRAPHSIZE*maxDegree*sizeof(int)); 
	memset(graphColors, 0, GRAPHSIZE*sizeof(int)); 
	memset(graphColors2, 0, GRAPHSIZE*sizeof(int)); 
	memset(boundaryList, 0, GRAPHSIZE*sizeof(int)); 
	memset(degreeList, 0, GRAPHSIZE*sizeof(int)); 
	

	
    long randSeed = time(NULL);
	srand ( randSeed );  // initialize random numbers  
	//srand ( 1271876520 );  // initialize random numbers   

	
	getAdjacentList(adjacencyMatrix, adjacencyList, GRAPHSIZE, maxDegree);
    getDegreeList(adjacencyList, degreeList, GRAPHSIZE, maxDegree);



	int iterations; 
	iterations= sdoIm(adjacencyList, graphColors, degreeList, GRAPHSIZE, maxDegree);
	numColors2 = firstFit(adjacencyMatrix, graphColors2, GRAPHSIZE, maxDegree);

	
	numColors = -1;
	for (int i=0; i<GRAPHSIZE; i++){
		if ( numColors < graphColors[i] )
			numColors = graphColors[i];
	}
/*
	// Display Graph
	cout << "Graph:" << endl;
	for (int i=0; i<GRAPHSIZE; i++){
		cout << i << " : ";
		for (int j=0; j<maxDegree; j++){
			cout << adjacencyList[i*maxDegree + j] << "  ";
		}
		cout << endl;
	}

	// Display Degree
	cout << endl << endl << "Degree:" << endl;
	for (int i=0; i<GRAPHSIZE; i++){
		cout << degreeList[i] << "  ";
	}
*/	
	// Graph info:
	cout << endl << endl << "Graph info:" << endl;
	cout << "Vertices: " << GRAPHSIZE << "   Edges: " << NUMEDGES << "   Density: " << 
		(2*NUMEDGES)/((float)GRAPHSIZE*(GRAPHSIZE-1))<< "   Degree: " << maxDegree << endl;
	
	
	cout <<  endl << "SDO check:";
	checkCorrectColoring(adjacencyMatrix, graphColors);
	
	cout << endl << "First Fit check:";
	checkCorrectColoring(adjacencyMatrix, graphColors2);
	
/*	
	// Display Colors - First fit
	cout << endl << endl << "SDO Colors:" << endl;
	for (int i=0; i<GRAPHSIZE; i++){
		cout << graphColors[i] << "  ";
		
	}
	
	// Display Colors - SDO improved
	cout << endl << endl << "First Fit Colors:" << endl;
	for (int i=0; i<GRAPHSIZE; i++){
		cout << graphColors2[i] << "  ";
		
	}
*/	
		
	cout << endl << "SDO Number of colors: " << numColors << "  iterations: " << iterations << endl;
	cout << "FF  Number of colors: " << numColors2 << endl;

	return 0;
}
