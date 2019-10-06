#include<iostream>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include "device_launch_parameters.h"
#include "device_functions.h"


// 32*32 image chunk + padding
/*
__global__ void reduce0(int *g_idata, int *g_odata) {
	extern __shared__ int sdata[1024];
	unsigned int tid = threadIdx.x;
	unsigned int i = threadIdx.x;
	// blockDim: num of threads in block in each direction
	sdata[tid] = g_idata[i];
	// Local shared memory to global memory mapping
	__syncthreads();

	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	
}
*/


namespace gpu
{
	__constant__ const int H_KERNEL_3 = 1;
	__constant__ const int H_KERNEL_5 = 2;
	__constant__ const int V_KERNEL_3 = 3;
	__constant__ const int V_KERNEL_5 = 4;
	__constant__ const int GAUSSIAN_KERNEL_3 = 5;
	__constant__ const int GAUSSIAN_KERNEL_5 = 6;

	__constant__ const int SHARD_SIZE = cpu::SHARD_SIZE;

    template <typename T>
    __device__ T get(T *dense, int x, int y, int rows, int cols)
    {
        /*
        Indexing (x,y) element in a dense array
    
        */
        int idx = x * cols + y;
        return dense[idx];
    }
    template <typename T>
    __device__ void set(T *dense, T value, int x, int y, int rows, int cols)
    {
        /*
        Setting (x, y) element in a dense array
        */
        int idx = x * cols + y;
        dense[idx] = value;
    }

    template <typename T>
    __device__ void abs(T* val) {
        if (*val<0) *val = -*val;
    }

	__device__ float gaussian(int x, int y, int mu, int sigma) {
		float pi = 3.1415926;
		float factor = 1 / (2 * pi * sigma * sigma);
		float expnt = -(x * x + y * y) / (2 * sigma * sigma);
		return factor * std::exp(expnt);
	}

	template <typename T>
	__device__ void get_kernel(int type, T** kernel, int* kernel_size) {
		int size;
		switch (type)
		{
		case H_KERNEL_3:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(int);
			//cudaMalloc((void**)kernel, size);
			*kernel = (int*)malloc(size);
			(*kernel)[0] = 1; (*kernel)[1] = 1; (*kernel)[2] = 1;
			(*kernel)[3] = 0; (*kernel)[4] = 0; (*kernel)[5] = 0;
			(*kernel)[6] = -1; (*kernel)[7] = -1; (*kernel)[8] = -1;
			break;

		case V_KERNEL_3:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(int);
			//cudaMalloc((void**)kernel, size);
			*kernel = (int*)malloc(size);
			(*kernel)[0] = 1; (*kernel)[1] = 0; (*kernel)[2] = -1;
			(*kernel)[3] = 1; (*kernel)[4] = 0; (*kernel)[5] = -1;
			(*kernel)[6] = 1; (*kernel)[7] = 0; (*kernel)[8] = -1;
			break;
		}
	}

	template <typename T>
	__device__ void normalize(T* arr, int size) {
		T sum;
		for (int i = 0; i < size; i++) {
			sum += arr[i];
		}
		for (int i = 0; i < size; i++) {
			arr[i] = arr[i] / sum;
		}
	}

	__device__ void get_gaussian_kernel(float** kernel, int kernel_size, int mu=0, int sigma=1) {
		int size;
		size = kernel_size * kernel_size * sizeof(float);
		*kernel = (float*)malloc(size);
		int center_x = (kernel_size-1) / 2;
		int center_y = (kernel_size-1) / 2;
		int dx, dy;
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				dx = center_x - i;
				dy = center_y - j;
				printf("%d", dx);
				float val = gaussian(dx, dy, mu, sigma);
				gpu::set(*kernel, val, i, j, kernel_size, kernel_size);
				
			}
		}

	}




	
}




