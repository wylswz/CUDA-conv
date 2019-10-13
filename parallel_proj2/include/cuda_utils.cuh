#include<iostream>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include "device_launch_parameters.h"
#include "device_functions.h"


namespace gpu
{
	__constant__ const int H_KERNEL_3 = cpu::H_KERNEL_3;
	__constant__ const int H_KERNEL_5 = cpu::H_KERNEL_5;
	__constant__ const int V_KERNEL_3 = cpu::V_KERNEL_3;
	__constant__ const int V_KERNEL_5 = cpu::V_KERNEL_3;
	__constant__ const int GAUSSIAN_KERNEL = cpu::GAUSSIAN_KERNEL;
	__constant__ const int DIAG_KERNEL_1 = cpu::DIAG_KERNEL_1;
	__constant__ const int DIAG_KERNEL_2 = cpu::DIAG_KERNEL_2;
	__constant__ const int SOBEL_H = cpu::SOBEL_H;
	__constant__ const int SOBEL_V = cpu::SOBEL_V;
	__constant__ const int SOBEL_H_5 = cpu::SOBEL_H_5;
	__constant__ const int SOBEL_V_5 = cpu::SOBEL_V_5;




	__constant__ const int IMG_COMB_ADD = cpu::IMG_COMB_ADD;
	__constant__ const int IMG_COMB_MAX = cpu::IMG_COMB_MAX;
	__constant__ const int IMG_COMB_MIN = cpu::IMG_COMB_MIN;
	__constant__ const int IMG_COMB_MEAN = cpu::IMG_COMB_MEAN;
	__constant__ const int IMG_COMB_MAGNITUDE = cpu::IMG_COMB_MAGNITUDE;


	__constant__ const int SHARD_SIZE = cpu::SHARD_SIZE;
	__constant__ const int PER_THREAD_SIZE = cpu::PER_THREAD_SIZE;

	template <typename T>
	__device__ T get(T* dense, int x, int y, int rows, int cols)
	{
		/*
		Indexing (x,y) element in a dense array

		*/
		int idx = x * cols + y;
		return dense[idx];
	}
	template <typename T>
	__device__ void set(T* dense, T value, int x, int y, int rows, int cols)
	{
		/*
		Setting (x, y) element in a dense array
		*/
		int idx = x * cols + y;
		dense[idx] = value;
	}

	__device__ float gaussian(int x, int y, float mu, float sigma) {
		float pi = 3.1415926;
		float factor = 1 / (2 * pi * sigma * sigma);
		float expnt = -(x * x + y * y) / (2 * sigma * sigma);

		return factor * expf(expnt);
	}

	template <typename T>
	__device__ void get_kernel(int type, T** kernel, int* kernel_size) {
		int size;
		switch (type)
		{
		case H_KERNEL_3:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = 1; (*kernel)[1] = 1; (*kernel)[2] = 1;
			(*kernel)[3] = 0; (*kernel)[4] = 0; (*kernel)[5] = 0;
			(*kernel)[6] = -1; (*kernel)[7] = -1; (*kernel)[8] = -1;
			break;

		case V_KERNEL_3:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = 1; (*kernel)[1] = 0; (*kernel)[2] = -1;
			(*kernel)[3] = 1; (*kernel)[4] = 0; (*kernel)[5] = -1;
			(*kernel)[6] = 1; (*kernel)[7] = 0; (*kernel)[8] = -1;
			break;
		case DIAG_KERNEL_1:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = 1; (*kernel)[1] = 1; (*kernel)[2] = 0;
			(*kernel)[3] = 1; (*kernel)[4] = 0; (*kernel)[5] = -1;
			(*kernel)[6] = 0; (*kernel)[7] = -1; (*kernel)[8] = -1;
			break;
		case DIAG_KERNEL_2:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = 0; (*kernel)[1] = 1; (*kernel)[2] = 1;
			(*kernel)[3] = -1; (*kernel)[4] = 0; (*kernel)[5] = 1;
			(*kernel)[6] = -1; (*kernel)[7] = -1; (*kernel)[8] = 0;
			break;

		case SOBEL_V:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = -1; (*kernel)[1] = 0; (*kernel)[2] = 1;
			(*kernel)[3] = -2; (*kernel)[4] = 0; (*kernel)[5] = 2;
			(*kernel)[6] = -1; (*kernel)[7] = 0; (*kernel)[8] = 1;
			break;

		case SOBEL_H:
			*kernel_size = 3;
			size = 3 * 3 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = -1; (*kernel)[1] = -2; (*kernel)[2] = -1;
			(*kernel)[3] = 0; (*kernel)[4] = 0; (*kernel)[5] = 0;
			(*kernel)[6] = 1; (*kernel)[7] = 2; (*kernel)[8] = 1;
			break;

		case SOBEL_H_5:
			*kernel_size = 5;
			size = 5 * 5 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = -1; (*kernel)[1] = -2; (*kernel)[2] = -1; (*kernel)[3] = 0; (*kernel)[4] = 0;
			(*kernel)[5] = -1; (*kernel)[6] = -2; (*kernel)[7] = -1; (*kernel)[8] = 0; (*kernel)[9] = 0;
			(*kernel)[10] = -1; (*kernel)[11] = -2; (*kernel)[12] = -1; (*kernel)[13] = 0; (*kernel)[14] = 0;
			(*kernel)[15] = -1; (*kernel)[16] = -2; (*kernel)[17] = -1; (*kernel)[18] = 0; (*kernel)[19] = 0;
			(*kernel)[20] = -1; (*kernel)[21] = -2; (*kernel)[22] = -1; (*kernel)[23] = 0; (*kernel)[24] = 0;
			break;

		case SOBEL_V_5:
			*kernel_size = 5;
			size = 5 * 5 * sizeof(T);
			*kernel = (T*)malloc(size);
			(*kernel)[0] = -1; (*kernel)[1] = -2; (*kernel)[2] = -1; (*kernel)[3] = 0; (*kernel)[4] = 0;
			(*kernel)[5] = -1; (*kernel)[6] = -2; (*kernel)[7] = -1; (*kernel)[8] = 0; (*kernel)[9] = 0;
			(*kernel)[10] = -1; (*kernel)[11] = -2; (*kernel)[12] = -1; (*kernel)[13] = 0; (*kernel)[14] = 0;
			(*kernel)[15] = -1; (*kernel)[16] = -2; (*kernel)[17] = -1; (*kernel)[18] = 0; (*kernel)[19] = 0;
			(*kernel)[20] = -1; (*kernel)[21] = -2; (*kernel)[22] = -1; (*kernel)[23] = 0; (*kernel)[24] = 0;
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

	__device__ void get_gaussian_kernel(float** kernel, int kernel_size, float mu = 0, float sigma = 1) {
		int size;
		size = kernel_size * kernel_size * sizeof(float);
		*kernel = (float*)malloc(size);
		int center_x = (kernel_size - 1) / 2;
		int center_y = (kernel_size - 1) / 2;
		int dx, dy;
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				dx = center_x - i;
				dy = center_y - j;
				float val = gaussian(dx, dy, mu, sigma);
				gpu::set(*kernel, val, i, j, kernel_size, kernel_size);

			}
		}

	}
	__device__ void get_gaussian_kernel(float kernel[], int kernel_size, float mu = 0, float sigma = 1) {

		int center_x = (kernel_size - 1) / 2;
		int center_y = (kernel_size - 1) / 2;
		int dx, dy;
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				dx = center_x - i;
				dy = center_y - j;
				float val = gaussian(dx, dy, mu, sigma);
				gpu::set(kernel, val, i, j, kernel_size, kernel_size);
				//printf("%f", val);

			}
		}

	}
}




