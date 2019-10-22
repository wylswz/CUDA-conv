#include<iostream>
#include <ctime>
#include<stdlib.h>
#include<string.h>
#include <cuda_texture_types.h>

#include "include/cv_utils.h"
#include"include/cuda_utils.cuh"
#include "include/cuda_kernels.cuh"
#include "include/EdgeFunctions.hpp"

texture <int, 1> tex_img;


void launch_conv_kernel(int* img, int rows, int cols, int* res, int ntests = 100) {

	int img_size = sizeof(int) * rows * cols;
	int* img_gpu;
	int* res_gpu[10];

	dim3 grid((rows / (cpu::SHARD_SIZE * cpu::PER_THREAD_SIZE)) + 1, (cols / (cpu::SHARD_SIZE * cpu::PER_THREAD_SIZE)) + 1, 1);
	dim3 block(cpu::SHARD_SIZE, cpu::SHARD_SIZE, 1);

	int block_x = cpu::SHARD_SIZE;
	int block_y = block_x;
	dim3 block_flat(block_x , block_y, 1);
	dim3 grid_flat((rows / block_x), (cols/block_y), 1);
	cudaMalloc((void**)&img_gpu, img_size);
	cudaMemcpy(img_gpu, img, img_size, cudaMemcpyHostToDevice);
	for (int i = 0; i < 4; i++) {
		cudaMalloc((void**)&res_gpu[i], img_size);
	}

	//const cudaChannelFormatDesc channelDesc =
	//	cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	
	//cudaBindTextureToArray(tex_img, cuimg, channelDesc);
	//cudaBindTexture(NULL, tex_img, img_gpu, channelDesc, img_size);
	for (int i = 0; i < ntests; i++) {
	convolution_2<cpu::GAUSSIAN_KERNEL> << < grid_flat, block_flat >> > (img_gpu, rows, cols, res_gpu[0]);
	cudaDeviceSynchronize();
	convolution_2<cpu::SOBEL_H> << < grid_flat, block_flat >> > (res_gpu[0], rows, cols, res_gpu[1]);
	convolution_2<cpu::SOBEL_V> << < grid_flat, block_flat >> > (res_gpu[0], rows, cols, res_gpu[2]);
	cudaDeviceSynchronize();
	image_combination << < grid_flat, block_flat >> > (res_gpu[3], res_gpu[1], res_gpu[2], rows, cols, cpu::IMG_COMB_MAGNITUDE);
	cudaMemcpy(res, res_gpu[3], img_size, cudaMemcpyDeviceToHost);
	}


	// Fetch result from GPU

	cudaFree(img_gpu);
	for (int i = 0; i < 4; i++) {
		cudaFree(res_gpu[i]);
	}

}



void edge_dection_cuda(char* path) {
	int rows = 64; int cols = 64;
	int* img = NULL;
	cpu::imread_dense(path, &img, IMREAD_GRAYSCALE, &rows, &cols);
	// Read image into a 1D array

	int img_size = rows * cols * sizeof(int);
	int* res = (int*)malloc(img_size);
	// Initialize result

	

	init_kernels << <1, 1 >> > ();
	malloc_gaussian_kernel << <1, 1 >> > ();
	cudaDeviceSynchronize();

	dim3 block(5, 5, 1);
	dim3 grid(gaussian_kernel_size / 5 + 1, gaussian_kernel_size / 5 + 1, 1);
	init_gaussian_kernel << <grid, block >> > ();
	cudaDeviceSynchronize();

	clock_t time_req;
	time_req = clock();
	launch_conv_kernel(img, rows, cols, res, 10);
	time_req = clock() - time_req;
	std::cout << "Speed: " << (float)time_req / CLOCKS_PER_SEC / 10 << " seconds per image" << std::endl;

	cpu::imshow(res, rows, cols, "None", 800, 800);
	free(res);
}




int main(int argc, char* argv[]) {

	char* my_path;
	if (argc > 1) {
		my_path = argv[1];
	}
	else {
		my_path = cpu::img_path;
	}

	edge_dection_cuda(my_path);
	//EdgeDetect(cpu::img_path);
}
