#include "vectorOp.h"


//Vector addition: C = a + b
__global__ void vectorAdd(float *cD, float *aD, float *bD, int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N)
		cD[i] = aD[i] + bD[i];	
} 




//Vector addition: C = a * b
__global__ void vectorMul(float *cD, float *aD, float *bD, int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N)
		cD[i] = aD[i] * bD[i];	
} 




extern "C" __host__ void vectorAddMulPrep(float *c, float *a, float *b, int N, int choice){
	float *cD, *aD, *bD;

	// Allocating memory on device
	cudaMalloc((void**)&cD, N * sizeof(float));
	cudaMalloc((void**)&aD, N * sizeof(float));
	cudaMalloc((void**)&bD, N * sizeof(float));

	
	// transfer data - a & b - from host(CPU) to device(GPU)
	cudaMemcpy(aD, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bD, b, N * sizeof(float), cudaMemcpyHostToDevice);

	
	//invoke function on GPU
	if (choice == 0)
		//launchAddKernel(cD, aD, bD, N);
		vectorAdd<<<ceil(N/(float)BLOCK_SIZE),BLOCK_SIZE>>>(cD, aD, bD, N);
	else
		//launchMulKernel(cD, aD, bD, N);
		vectorMul<<<ceil(N/(float)BLOCK_SIZE),BLOCK_SIZE>>>(cD, aD, bD, N);


	// transfer result of addition - c - from device(GPU) to host(CPU)
	cudaMemcpy(c, cD, N * sizeof(float), cudaMemcpyDeviceToHost);


	// free memory
	cudaFree(cD);
	cudaFree(aD);
	cudaFree(bD);
}
