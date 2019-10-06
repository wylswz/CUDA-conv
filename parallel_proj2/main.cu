#include<iostream>
#include <ctime>
#include<stdlib.h>
#include<string.h>

#include "include/cv_utils.h"
#include"include/cuda_utils.cuh"
#include "include/cuda_kernels.cuh"



void launch_conv_kernel(int* img, int rows, int cols, int* kernel, int kernel_dim,  int* res) {

	int img_size = sizeof(int) * rows * cols;
	int* kernel_gpu;
	int* img_gpu;
	int* res_gpu;
	
	// int kernel_size = sizeof(int) * kernel_dim * kernel_dim;

	dim3 grid((rows / (cpu::SHARD_SIZE - 2) + 1), (cols / (cpu::SHARD_SIZE - 2) + 1), 1);
	dim3 block(cpu::SHARD_SIZE - 2, cpu::SHARD_SIZE - 2, 1);

	// cudaMalloc((void**)&kernel_gpu, kernel_size);
	cudaMalloc((void**)&img_gpu, img_size);
	//cudaMemcpy(kernel_gpu, kernel, kernel_size, cudaMemcpyHostToDevice);
	cudaMemcpy(img_gpu, img, img_size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&res_gpu, img_size);
	// Copy kernel, result and image to gpu memory
	
	init_kernels <<<1,1>>> ();
	cudaDeviceSynchronize();
	
	//convolution_0 <<< grid, block >>> (img_gpu, rows, cols, kernel_gpu, kernel_dim, res_gpu);
	convolution_1 <<< grid, block >>> (img_gpu, rows, cols, res_gpu);
	cudaMemcpy(res, res_gpu, img_size, cudaMemcpyDeviceToHost);
	// Fetch result from GPU

	printf(" %d ", res[0]);

	cudaFree(img_gpu);
	//cudaFree(kernel_gpu);
	cudaFree(res_gpu);
}

std::vector<cv::Mat> batch_edge_detection(std::vector<cv::String> images_path) {



	std::vector <cv::Mat> res(images_path.size());
	return res;
}

int main(int argc, char* argv[]) {

	int rows; int cols;
	int* img = NULL;
	cpu::imread_dense(cpu::img_path, &img, IMREAD_GRAYSCALE,&rows, &cols, 4);
	// Read image into a 1D array

	int img_size = rows * cols * sizeof(int);
	int* res = (int*)malloc(img_size);
	// Initialize result

	int kernel_dim=0; int* kernel=NULL;
	cpu::get_kernel(cpu::H_KERNEL_3, &kernel, &kernel_dim);
	// get convolution kernel

	clock_t time_req;
	time_req = clock();
	launch_conv_kernel(img, rows, cols,kernel, kernel_dim, res);
	time_req = clock() - time_req;
	std::cout << "Speed: " << (float)time_req / CLOCKS_PER_SEC << " seconds per image" << std::endl;

	// Launch cuda kernel
	

	cpu::imshow(res, rows, cols);


	free(res);
	free(kernel);
	
}
