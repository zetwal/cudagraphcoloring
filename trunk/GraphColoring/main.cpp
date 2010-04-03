// Graph coloring

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
using namespace std;

const int GRAPHSIZE = 30;


void generateMatrix(int matrix[GRAPHSIZE][GRAPHSIZE], int num){
	int x, y;
	int count = 0;
	
	while (count < num){
		x = rand()%GRAPHSIZE;
		y = rand()%GRAPHSIZE;
		
		if (x != y){
			matrix[x][y] = 1;	// non directional graph
			matrix[y][x] = 1;
			count++;
		}
	}
}


int getMaxDegree(int adjacencyMatrix[GRAPHSIZE][GRAPHSIZE]){
	int maxDegree = 0; 
	int degree;
	
	for (int i=0; i<GRAPHSIZE; i++){
		degree = 0;
		
		for (int j=0; j<GRAPHSIZE; j++)		
			if (	adjacencyMatrix[i][j] == 1)
				degree++;
		
		if (degree > maxDegree)
			maxDegree = degree;
	}
	
	return maxDegree;
}




// First Fit - simplest one ever
int colorGraph(int matrix[GRAPHSIZE][GRAPHSIZE], int colors[GRAPHSIZE], int maxDegree){
	int numColors = 0;
	int i, j;
	
	int * degreeArray;
	degreeArray = new int[maxDegree+1];
	
	
	for (i=0; i<GRAPHSIZE; i++)
	{		
		// initialize degree array
		for (j=0; j<=maxDegree; j++)
			degreeArray[j] = j+1;
		
		
		// check the colors
		for (j=0; j<GRAPHSIZE; j++){
			if (i == j)
				continue;
			
			// check connected
			if (	matrix[i][j] == 1)
				if (colors[j] != 0)
					degreeArray[colors[j]-1] = 0;	// set connected spots to 0
		}
		
		for (j=0; j<=maxDegree; j++)
			if (degreeArray[j] != 0){
				colors[i] = degreeArray[j];
				break;
			}
		
		if (colors[i] > numColors)
			numColors=colors[i];
	}
	
	return numColors;
}






int main(){
	int adjacencyMatrix[GRAPHSIZE][GRAPHSIZE] = {0};		// initialize graph data
	int graphColors[GRAPHSIZE] = {0};
	int numColors = 0;
	int maxDegree;
	
	srand ( time(NULL) );								// initialize random numbers
	
	// initialize graph
	generateMatrix(adjacencyMatrix, 450);
	
	// Display graph
	for (int i=0; i<GRAPHSIZE; i++){
		for (int j=0; j<GRAPHSIZE; j++)
			cout << adjacencyMatrix[i][j] << "  ";
		
		cout << endl;
	}
	
	
	// determining the maximum degree
	maxDegree = getMaxDegree(adjacencyMatrix);
	cout << "Max degree: " << maxDegree << endl;
	
	
	// graph coloring
	numColors = colorGraph(adjacencyMatrix, graphColors, maxDegree);	
	
	for (int k=0; k<GRAPHSIZE; k++)
		cout << graphColors[k] << "  ";
	
	cout << endl;
	
	// display graph
	cout << "Number of colors: " << numColors << endl;
	
	return 0;
}

