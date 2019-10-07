#include<iostream>
#include <ctime>
#include<stdlib.h>
#include<string.h>

#include "include/cv_utils.h"
#include"include/cuda_utils.cuh"
#include "include/cuda_kernels.cuh"

void launch_conv_kernel(int* img, int rows, int cols, int* res) {

	int img_size = sizeof(int) * rows * cols;
	int* img_gpu;
	int* res_gpu[10];
	
	dim3 grid((rows / (cpu::SHARD_SIZE - 2) + 1), (cols / (cpu::SHARD_SIZE - 2) + 1), 1);
	dim3 block(cpu::SHARD_SIZE - 2, cpu::SHARD_SIZE - 2, 1);

	cudaMalloc((void**)&img_gpu, img_size);
	cudaMemcpy(img_gpu, img, img_size, cudaMemcpyHostToDevice);
	for (int i = 0; i < 10; i++) {
		cudaMalloc((void**)&res_gpu[i], img_size);
	}
	
	convolution_1 <<< grid, block >>> (img_gpu, rows, cols, res_gpu[0], cpu::GAUSSIAN_KERNEL);
	cudaDeviceSynchronize();
	convolution_1 <<< grid, block >>> (res_gpu[0], rows, cols, res_gpu[1], cpu::H_KERNEL_3);
	convolution_1 <<< grid, block >>> (res_gpu[0], rows, cols, res_gpu[2], cpu::V_KERNEL_3);
	convolution_1 << < grid, block >> > (res_gpu[0], rows, cols, res_gpu[3], cpu::DIAG_KERNEL_1);
	convolution_1 << < grid, block >> > (res_gpu[0], rows, cols, res_gpu[4], cpu::DIAG_KERNEL_2);
	cudaDeviceSynchronize();
	image_combination <<< grid, block >>> (res_gpu[5], res_gpu[1], res_gpu[2], rows, cols);
	image_combination << < grid, block >> > (res_gpu[5], res_gpu[5], res_gpu[3], rows, cols);
	image_combination << < grid, block >> > (res_gpu[5], res_gpu[5], res_gpu[4], rows, cols);
	cudaMemcpy(res, res_gpu[5], img_size, cudaMemcpyDeviceToHost);
	// Fetch result from GPU

	cudaFree(img_gpu);
	cudaFree(res_gpu);
}


int main(int argc, char* argv[]) {

	int rows; int cols;
	int* img = NULL;
	cpu::imread_dense(cpu::img_path, &img, IMREAD_GRAYSCALE,&rows, &cols, 4);
	// Read image into a 1D array

	int img_size = rows * cols * sizeof(int);
	int* res = (int*)malloc(img_size);
	// Initialize result

	init_kernels <<<1, 1 >>> ();
	malloc_gaussian_kernel << <1, 1 >> > ();
	cudaDeviceSynchronize();

	dim3 block(5, 5, 1);
	dim3 grid(gaussian_kernel_size / 5 + 1, gaussian_kernel_size / 5 + 1, 1);
	init_gaussian_kernel <<<grid, block >>> ();
	cudaDeviceSynchronize();

	clock_t time_req;
	time_req = clock();
	launch_conv_kernel(img, rows, cols, res);
	time_req = clock() - time_req;
	std::cout << "Speed: " << (float)time_req / CLOCKS_PER_SEC << " seconds per image" << std::endl;

	// Launch cuda kernel
	
	cpu::imshow(res, rows, cols,"None",800,800);

	free(res);	
}
