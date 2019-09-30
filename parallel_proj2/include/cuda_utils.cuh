#include<iostream>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <cuda.h>
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



	
}




__global__ void convolution_0(int* image, int img_rows, int img_cols, int* kernel,int kernel_size, int* image_out) {
    /*
    Access image data using slow global memory
    */
    //const int shard_size = blockDim.x;
    //const int shard_size_padded = blockDim.x + kernel_size/2;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sum=0;

    if (x<0 || x>=img_rows || y<0 || y>=img_cols) {
        return;
    } else {
        for (int i = 0;i<kernel_size;i++) {
            for (int j=0;j<kernel_size; j++) {
                int temp_x = x - kernel_size/2 + i;
                int temp_y = y - kernel_size/2 + j;
                // temp_x and temp_y are index corresponding to 
                // cells in convolution kernel
                if (temp_x <0 || temp_x >=img_rows || temp_y < 0 || temp_y>img_cols) {
                    sum += 0;
                } else {
                    sum += gpu::get(image,temp_x, temp_y, img_rows, img_cols)*\
                    gpu::get(kernel, i, j, kernel_size, kernel_size);
					gpu::abs(&sum);
                }   
            }
        }
        //__syncthreads();
        gpu::set(image_out, sum, x, y, img_rows, img_cols);
    }
	__syncthreads();
}

__global__ void convolution_1(int* image, int img_rows, int img_cols, int* kernel,int kernel_size, int* image_out) {
    /*
    Attempt 1: Try to copy image chunk to shared memory for each block
    */
    //const int shard_size = 32;
    //const int shard_size_padded = 32 + kernel_size/2;
    extern __shared__ int img_chunk[gpu::SHARD_SIZE * gpu::SHARD_SIZE];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    

    int sum=0;

    if (x<0 || x>=img_rows || y<0 || y>=img_rows) {
        return;
    } else {
        
    }
}