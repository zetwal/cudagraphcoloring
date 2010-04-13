#ifndef _VECTOROP_H_
#define _VECTOROP_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>


#define BLOCK_SIZE 100

#ifdef __cplusplus
	#define CHECK_EXT extern "C"
#else
	#define CHECK_EXT
#endif

CHECK_EXT __host__ void vectorAddMulPrep(float *c, float *a, float *b, int N, int choice);



#endif // _VECTOROP_H_
