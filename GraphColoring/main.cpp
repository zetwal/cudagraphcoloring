// Graph coloring 
  
 #include <stdio.h> 
 #include <stdlib.h> 
 #include <time.h> 
 #include <iostream>
 #include <math.h> 
 using namespace std; 
  
const int GRAPHSIZE = 30; 
const int SUBSIZE = 10;
  
  
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
	if(n < numSub-1){	
	    width = subsize;
	}
	else{
	    width = size - (numSub-1)*subsize;
	}

		
	for(int j = 0; j < width; j++)
	    for(int i = 0; i < width; i++)
	        {
		    int ii = i + n*subsize;
		    int jj = j + n*subsize;
		
		    subMatrix[i*subsize + j] = matrix[ii*size + jj];
	        }

} 
  
  
 int main(){ 
        int *adjacencyMatrix = new int[GRAPHSIZE*GRAPHSIZE*sizeof(int)]; 
	int *graphColors = new int[GRAPHSIZE*sizeof(int)]; 
        // initialize graph data 
        
         int numColors = 0; 
         int maxDegree; 
          
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
          
         cout<<"global graph coloring results:"<<endl;
         for (int k=0; k<GRAPHSIZE; k++) 
                 cout << graphColors[k] << "  "; 
          
         cout << endl; 
          
         // display graph 
         cout << "Number of colors: " << numColors << endl; 


         // partitioning
         int numSub = ceil(double(GRAPHSIZE)/double(SUBSIZE));
         int *subMatrix = new int[SUBSIZE*SUBSIZE*sizeof(int)];
	     int *subgraphColors = new int[SUBSIZE*sizeof(int)];      
	 
         int i,j,k,n;
         for(i=0; i<numSub; i++)
	 {
	     getSubMatrix(adjacencyMatrix, subMatrix, GRAPHSIZE, SUBSIZE, i, numSub);
	 	
		// Display graph 
             for (int i1=0; i1<SUBSIZE; i1++){ 
                 for (j=0; j<SUBSIZE; j++) 
                         cout << subMatrix[i1*SUBSIZE + j] << "  "; 
                  
                 cout << endl; 
             }
	     maxDegree = getMaxDegree(subMatrix, SUBSIZE); 
             cout << "Max degree of subMatrix: " << maxDegree << endl; 
	     numColors = colorGraph(subMatrix, subgraphColors, SUBSIZE, maxDegree);

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

         int *conflict = new int[GRAPHSIZE*sizeof(int)];
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
         
	 
	 cout<<"number of conflicting nodes ="<<conflictCount<<endl;
	
         for (k=0; k<GRAPHSIZE; k++) 
                 cout << conflict[k] << "  "; 
          
         cout << endl; 
	
	 for (i=0; i<GRAPHSIZE; i++) 
	{
	    
	}
	
	delete[] adjacencyMatrix;
	delete[] graphColors;
	delete[] subMatrix;
	delete[] subgraphColors;
	delete[] conflict;
         return 0; 
 } 
 
