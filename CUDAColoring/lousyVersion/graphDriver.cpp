// Graph coloring 


#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <iostream>
#include <math.h> 
#include <set>
#include "graphColoring.h"

using namespace std; 





#define UNIQUE_CONFLICT


int   solveConflict(int *matrix, int size, int *conflict, int conflictSize, int *graphColors)
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


void generateMatrix(int *matrix, int size, int num){ 
	int x, y; 
	int count = 0; 
	cout<<"num="<<num<<endl; 
	cout<<"enter generateMatrix()"<<endl;
	while (count < num){ 
		x = rand()%GRAPHSIZE; 
		y = rand()%GRAPHSIZE; 
		//cout<<"x="<<x<<endl;
		//cout<<"y="<<y<<endl;
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


// First Fit - simplest one ever 
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

void getSubMatrix(int *matrix, int *subMatrix, int size, int subsize, int n, int numSub)
{
	int width;
	if (n < numSub-1){	
	    width = subsize;
	}
	else{
	    width = size - (numSub-1)*subsize;
	}
	
	
	for (int j = 0; j < width; j++)
	    for (int i = 0; i < width; i++)
		{
		    int ii = i + n*subsize;
		    int jj = j + n*subsize;
			
		    subMatrix[i*subsize + j] = matrix[ii*size + jj];
		}
	
} 


int getConflictedNodes(int *adjacencyMatrix, int *graphColors, int *conflict)
{
	// partitioning
	int numSub = ceil((float)GRAPHSIZE/(float)SUBSIZE);
	int *subMatrix = new int[SUBSIZE*SUBSIZE*sizeof(int)];
	int *subgraphColors = new int[SUBSIZE*sizeof(int)];   
	memset(subMatrix, 0, SUBSIZE*SUBSIZE*sizeof(int)); 
	memset(subgraphColors, 0, SUBSIZE*sizeof(int));  
	
	int i,j,k,n, maxDegree, numColors, maxColor = 1;
	for(i=0; i<numSub; i++)
	{
		getSubMatrix(adjacencyMatrix, subMatrix, GRAPHSIZE, SUBSIZE, i, numSub);
	 	
		// Display subMatrix 
		for (int i1=0; i1<SUBSIZE; i1++){ 
			for (j=0; j<SUBSIZE; j++) 
				cout << subMatrix[i1*SUBSIZE + j] << "  "; 
			
			cout << endl; 
		}
		
		maxDegree = getMaxDegree(subMatrix, SUBSIZE);
		
		cout << "Max degree of subMatrix: " << maxDegree << endl; 
		numColors = colorGraph(subMatrix, subgraphColors, SUBSIZE, maxDegree);
		
		if(maxColor < numColors)
		{
			maxColor = numColors;
		}	
		
		for(int k=0; k<SUBSIZE; k++)
		{
			int kk = i*SUBSIZE + k;
			graphColors[kk] = subgraphColors[k];
		}     
	}
	
	cout<<"partitioned graphColors:"<<endl;	 
	for (k=0; k<GRAPHSIZE; k++) 
		cout << graphColors[k] << "  "; 
	
	cout << endl; 
	
	cout<<"number of colors:"<<maxColor<<endl;
	
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
					conflict[conflictCount] = min(ii,j) + 1;
					conflictCount++;
				}
			}		
		}
		
	}
	
	delete[] subMatrix;
	delete[] subgraphColors;
	
	int * newConflict = new int[GRAPHSIZE];
	for(int i=0; i<GRAPHSIZE; i++)
		newConflict[i] = conflict[i];
	
    cout<<"List of conflicting nodes:"<<endl;
    for (int k=0; k<conflictCount; k++) 
		cout << conflict[k] << "  "; 	
	cout << "\n";
	
	//unique conflict 
#ifdef UNIQUE_CONFLICT	
	bool repeat = false;
	int count = 0;
	for(int i=0; i<GRAPHSIZE; i++)
	{
		repeat = false;
		for(int j=0; j<i; j++)
			if(conflict[i] == conflict[j] || conflict[i] == 0)
				repeat = true;
		if(!repeat)
			count++;
		
	}
	
    conflictCount = count;
	
	count = 0;
	for(int i=0; i<GRAPHSIZE; i++)
	{
		repeat = false;
		for(int j=0; j<i; j++)
			if(conflict[i] == conflict[j] || conflict[i] == 0)
				repeat = true;
		if(!repeat)
		{
			conflict[count] = newConflict[i];
			count++;
		}
		
	}
	
	delete []  newConflict;
	
#endif
	
	return conflictCount;
}





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
    		cout<<"List of conflicting nodes:"<<endl;
        for (int i=0; it != conflictSet.end(); it++) 
        {
                conflict[i] = *it;
                i++;
                cout << conflict[i] << "  ";    
        }
        cout << "\n";
        

        return conflictSet.size();

}


void checkCorrectColoring(int *adjacencyMatrix, int *graphColors){
	int numErrors = 0;
	
	cout << endl << "==================" << endl << "Error checking for Graph" << endl;
	
	for (int i=0; i<GRAPHSIZE; i++){	// we check each row
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
	// initialize graph data 
	memset(adjacencyMatrix, 0, GRAPHSIZE*GRAPHSIZE*sizeof(int));
	memset(graphColors, 0, GRAPHSIZE*sizeof(int));
	
	int numColorsSeq, numColorsParallel;
	int numColors = 0; 
	int maxDegree; 
	numColorsSeq = numColorsParallel = 0;

	
	srand ( time(NULL) );                                                           // initialize random numbers 
	
	// initialize graph 
	generateMatrix(adjacencyMatrix, GRAPHSIZE, 100); 
	
	// Display graph 
	for (int i=0; i<GRAPHSIZE; i++){ 
		for (int j=0; j<GRAPHSIZE; j++) 
			cout << adjacencyMatrix[i*GRAPHSIZE + j] << "  "; 
		
		cout << endl; 
	} 
	
	
	// determining the maximum degree 
	maxDegree = getMaxDegree(adjacencyMatrix, GRAPHSIZE); 
	cout << "Max degree: " << maxDegree << endl; 
	
	
	// graph coloring 
	numColors = colorGraph(adjacencyMatrix, graphColors, GRAPHSIZE, maxDegree);     
	numColorsSeq = numColors;
	
	cout<<"Number of colors:"<<numColors<<endl;    
	
	cout<<"global graph coloring results:"<<endl;
	for (int k=0; k<GRAPHSIZE; k++) 
		cout << graphColors[k] << "  "; 
	cout << endl; 
	
	// Checking for color conflict
	checkCorrectColoring(adjacencyMatrix, graphColors);
	cout << endl; 
	
	
	// partition and color subgraphs
	int *conflict = new int[GRAPHSIZE*sizeof(int)];
	memset(conflict, 0, GRAPHSIZE*sizeof(int));
	memset(graphColors, 0, GRAPHSIZE*sizeof(int));  

	//int conflictCount = getConflictedNodes(adjacencyMatrix, graphColors, conflict);
	subGraphColoring(adjacencyMatrix, graphColors, maxDegree);
	int conflictCount = getConflicts(adjacencyMatrix, graphColors, conflict);
	
	cout<<"List of conflicting nodes:"<<endl;
	for (int k=0; k<conflictCount; k++) 
		cout << conflict[k] << "  "; 
	cout << endl; 
	
	
    //solve the conflicts	
	//original adjacencyMatrix 
	//conflict nodes
	//check each nodes to find the right color
	//get new color 
    numColors = solveConflict(adjacencyMatrix,  GRAPHSIZE, conflict, conflictCount, graphColors);
	numColorsParallel = numColors;
	
	
	
	for (int i=0; i<GRAPHSIZE; i++){ 
		for (int j=0; j<GRAPHSIZE; j++) 
			cout << adjacencyMatrix[i*GRAPHSIZE + j] << "  "; 
		
		cout << endl; 
	} 	
	cout<<"global graph coloring results:"<<endl;
	for (int k=0; k<GRAPHSIZE; k++) 
		cout << graphColors[k] << "  "; 
	cout << endl;
	
    cout <<"Number of colors after solve conflict:" << numColors << endl;    

	cout << endl << "Sequential Colors: " << numColorsSeq << " 	- 	Prarallel Colors: " << numColorsParallel << endl;    
	
	
	// Checking for color conflict
	checkCorrectColoring(adjacencyMatrix, graphColors);
	
	
	delete[] adjacencyMatrix;
	delete[] graphColors;
	delete[] conflict;
	
	return 0; 
} 
