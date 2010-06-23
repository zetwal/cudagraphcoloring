#include <ctime> 
#include <stdio.h>  
#include <stdlib.h>  
#include <time.h>  
#include <iostream> 
#include <math.h>  
#include <set> 
#include <assert.h>
#include <fstream>
#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 

#include <iostream>
using namespace std;

int main(int argc, char* argv[]){
	if (argc != 3){
		cout << "2 arguments needed!" << endl;
		cout << "exe inputFilename weighted(0: weighted graph, 1: unweighted)" << endl;
	}
	ifstream graphFile(argv[1]);
	
	bool weighted;
	char comments[512];
	int graphSizeX, graphSizeY, numEdges, numUsefulEdges;
	unsigned int from, to;
	float weight;
	numUsefulEdges = 0;
	
	int *adjacencyMatrix;
	
	if (atoi(argv[2]) == 0)
		weighted = true;
	else	
		weighted = false;
	
	
	if (graphFile.is_open())
	{
		while (graphFile.peek() == '%'){
			graphFile.getline(comments,512);
		}
		
		graphFile >> graphSizeX >> graphSizeY >> numEdges;
		cout << graphSizeX << " " <<  graphSizeY << " - " << numEdges << endl;
		
		if (graphSizeX != graphSizeY){
			cout << "Non Symmetric graph!" << endl;
			exit(1);
		}
		else 
		{
			adjacencyMatrix = new int[graphSizeX * graphSizeX];
			memset(adjacencyMatrix, 0, graphSizeX * graphSizeX *sizeof(int));
			
			for (int i=0; i<numEdges; i++){
				if (weighted == true){
					graphFile >> from >> to >> weight;	
					cout << from << " , " << to << " : " << weight << endl;
				}
				else{
					graphFile >> from >> to;	
					cout << from << " , " << to << endl;
				}
					
				
				
				if (from != to){
					adjacencyMatrix[(from-1)*graphSizeX + (to-1)] = 1;
					adjacencyMatrix[(to-1)*graphSizeX + (from-1)] = 1;
					
					numUsefulEdges++;	
				}
			}
		}
	}
	else {
		cout << "Reading " << argv[1] << " failed!" << endl;
		exit(1);
	}
	
	int degree, maxDegree;
	maxDegree = degree = 0;
	
	for (int i=0; i<graphSizeX; i++){
		degree = 0;
		for (int j=0; j<graphSizeX; j++){
			if (adjacencyMatrix[i*graphSizeX + j] == 1)
				degree++;
		}
		
		if (maxDegree < degree)
			maxDegree = degree;
		
		cout << i << ": " << degree << endl;
	}
	cout << "Vertices: " << graphSizeX << endl;
	cout << "Num edges: " << numUsefulEdges << endl;
	cout << "Max degree: " << maxDegree << endl;
		
	
	delete []adjacencyMatrix;

	return 0;
}
