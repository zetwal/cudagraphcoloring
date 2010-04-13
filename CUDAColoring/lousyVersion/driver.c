#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "vectorOp.h"

int func_add ( float *x, float *y, int sz)
{
	int i;
	float *a;
	a = ( float *)malloc(sizeof(float)*sz);
	if (!a){
		printf("memory allocation error\n");
		exit(-1);
	}
	memcpy(a,x,sz*(sizeof(float)));

	/* replace the code to add
         * with a cuda call which you will
	 * implement as a interface to your cuda enabled library
	 */
	vectorAddMulPrep(x, a, y, sz, 0);

/*	for ( i=0; i<sz; i++)
		x[i]+=y[i];
*/
		
	for ( i=0; i<sz; i++){
		if (x[i]!= a[i] + y[i]){
			return 0;
		}
	}
		
	free(a);
	return 1;
}

	 	
int func_mul ( float *x, float *y, int sz)
{
	int i;
	float *a;
	a = ( float *)malloc(sizeof(float)*sz);
	if (!a){
		printf("memory allocation error\n");
		exit(-1);
	}
	memcpy(a,x,sz*(sizeof(float)));

	/* replace the code to multiply
         * with a cuda call which you will
	 * implement as a interface to your cuda enabled library
	 */

	vectorAddMulPrep(x, a, y, sz, 1);
	/*
	for ( i=0; i<sz; i++)
		x[i]*=y[i];
	*/

	
	for ( i=0; i<sz; i++){
		if (x[i]!= a[i] * y[i]){
			return 0;
		}
	}
	
	free(a);
	
	return 1;
}

int main()
{
	float *a,*b;
	int i, j;

	for ( j=10; j<1000000; j*=10){
		a =( float *) malloc(sizeof(float)*j);
		b =( float *) malloc(sizeof(float)*j);

		for (i=0; i<j; i++){
			a[i] = 2;
			b[i] = 3;
		}


		if(!func_add(a,b,j)){
			printf("failed to add\n");
		}
		else{
			printf("add operation completed\n");
		}
		

		if(!func_mul(a,b,j)){
			printf("failed to mul\n");
		}
		else{
			printf("mul operation completed\n");
		}	
		

		free(a);
		free(b);
	}

	return 0;

}
