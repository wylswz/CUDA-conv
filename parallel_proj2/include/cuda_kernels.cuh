
#define GAUSSIAN_KERNEL_SIZE 3


__device__ float* gaussian_kernel;
__device__ float* kernels[10];
__device__ int kernel_sizes[10];
const int gaussian_kernel_size = GAUSSIAN_KERNEL_SIZE;

__global__ void init_kernels() {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		gpu::get_kernel(gpu::H_KERNEL_3, &kernels[0], &kernel_sizes[0]);
		gpu::get_kernel(gpu::V_KERNEL_3, &kernels[1], &kernel_sizes[1]);
		gpu::get_kernel(gpu::DIAG_KERNEL_1, &kernels[2], &kernel_sizes[2]);
		gpu::get_kernel(gpu::DIAG_KERNEL_2, &kernels[3], &kernel_sizes[3]);
		gpu::get_kernel(gpu::SOBEL_H, &kernels[4], &kernel_sizes[4]);
		gpu::get_kernel(gpu::SOBEL_V, &kernels[5], &kernel_sizes[5]);
		gpu::get_kernel(gpu::SOBEL_H_5, &kernels[6], &kernel_sizes[6]);
		gpu::get_kernel(gpu::SOBEL_V_5, &kernels[7], &kernel_sizes[7]);
	}
	__syncthreads();
}

__global__ void malloc_gaussian_kernel() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x == 0 && y == 0) {
		int size = GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE * sizeof(float);
		gaussian_kernel = (float*)malloc(size);
	}
}


__global__ void init_gaussian_kernel(float mu = 0, float sigma = 1) {
	int dim = GAUSSIAN_KERNEL_SIZE;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dim || y >= dim) {

	}
	else {

		int centre_x = (dim - 1) / 2;
		int centre_y = (dim - 1) / 2;
		float res = gpu::gaussian(centre_x - x, centre_y - y, mu, sigma);
		gpu::set(gaussian_kernel, res, x, y, dim, dim);
	}
	__syncthreads();
}

__global__ void image_combination(int* dst, int* img1, int* img2, int rows, int cols, int method = gpu::IMG_COMB_MAX) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < 0 || x >= rows || y < 0 || y >= cols) {

	}
	else {
		int res = 0;
		switch (method)
		{
		case gpu::IMG_COMB_MAX:
			res = fminf(gpu::get(img1, x, y, rows, cols), gpu::get(img2, x, y, rows, cols));
			gpu::set(dst, res, x, y, rows, cols);
			break;
		case gpu::IMG_COMB_MAGNITUDE:
			res = hypotf(img1[x * cols + y], img2[x * cols + y]);
			gpu::set(dst, res, x, y, rows, cols);
			break;
		case gpu::IMG_COMB_MIN:
			res = fminf(gpu::get(img1, x, y, rows, cols), gpu::get(img2, x, y, rows, cols));
			gpu::set(dst, res, x, y, rows, cols);
			break;
		case gpu::IMG_COMB_MEAN:
			res = (gpu::get(img1, x, y, rows, cols) + gpu::get(img2, x, y, rows, cols)) / 2;
			gpu::set(dst, res, x, y, rows, cols);
			break;
		case gpu::IMG_COMB_ADD:
			res = gpu::get(img1, x, y, rows, cols) + gpu::get(img2, x, y, rows, cols);
			gpu::set(dst, res, x, y, rows, cols);
			break;
		default:
			break;
		}
	}
	__syncthreads();

}

__global__ void convolution_0(int* image, int img_rows, int img_cols, int* kernel, int kernel_size, int* image_out) {

	/*
	Initial implementation. Convolution masks are copied for each kernel
	*/

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sum = 0;

	if (x < 0 || x >= img_rows || y < 0 || y >= img_cols) {
		// Do Nothing
	}
	else {
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				int temp_x = x - kernel_size / 2 + i;
				int temp_y = y - kernel_size / 2 + j;

				if (temp_x < 0 || temp_x >= img_rows || temp_y < 0 || temp_y>img_cols) {
					//sum += 0;
				}
				else {
					sum += gpu::get(image, temp_x, temp_y, img_rows, img_cols) * \
						gpu::get(kernel, i, j, kernel_size, kernel_size);
				}
			}
		}
		sum = fabsf(sum);
		gpu::set(image_out, sum, x, y, img_rows, img_cols);
	}
	__syncthreads();
}

template <int kernel_idx>
__global__ void convolution_1(int* image, int img_rows, int img_cols, int* image_out) {
	/*
	pixel-per-thread implementation with shared-memory kernel
	and complete inner loop
	*/
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int kernel_size;
	float kernel[1024];
	int rows = img_rows;
	int cols = img_cols;


	switch (kernel_idx)
	{
	case gpu::SOBEL_H:
		kernel_size = 3;
		memcpy(kernel, kernels[4], kernel_size * kernel_size * sizeof(float));
		break;
	case gpu::SOBEL_V:
		kernel_size = 3;
		memcpy(kernel, kernels[5], kernel_size * kernel_size * sizeof(float));
		break;
	case gpu::GAUSSIAN_KERNEL:
		kernel_size = GAUSSIAN_KERNEL_SIZE;
		memcpy(kernel, gaussian_kernel, kernel_size * kernel_size * sizeof(float));
		break;


	default:
		break;
	}

	__syncthreads();

	int sum = 0;

	if (x >= 0 && x < rows && y >= 0 && y < cols) {
		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				int temp_x = x - kernel_size / 2 + i;
				int temp_y = y - kernel_size / 2 + j;
				if (temp_x < 0 || temp_x >= rows || temp_y < 0 || temp_y>cols) {
					sum += 0;
				}
				else {

					sum += image[temp_x * cols + temp_y] * \
						kernel[i * kernel_size + j];
				}
			}
		}

		sum = fabsf(sum);
		gpu::set(image_out, sum, x, y, rows, cols);
	}

	__syncthreads();
}


template <int kernel_idx>
__global__ void convolution_2(int* image, int img_rows, int img_cols, int* image_out) {
	/*
		- pixel-per-thread implementation with
		 global kernel access and unrolled inner loop
		- Reduced branch by dropping pixels on the edge of image
	*/

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int kernel_size;
	float* kernel;
	switch (kernel_idx)
	{
	case gpu::SOBEL_H:
		kernel_size = 3;
		kernel = kernels[4];
		break;
	case gpu::SOBEL_V:
		kernel_size = 3;
		kernel = kernels[5];
		break;
	case gpu::GAUSSIAN_KERNEL:
		kernel_size = GAUSSIAN_KERNEL_SIZE;
		kernel = gaussian_kernel;
		break;

	default:
		break;
	}
	__syncthreads();


	if (x >= 1 && x < img_rows - 1 && y >= 1 && y < img_cols - 1) {
		/*for (unsigned int i = 0; i < kernel_size; i++) {
			for (unsigned int j = 0; j < kernel_size; j++) {
				temp_x = __usad(x, half_kernel_size, i);//x - half_kernel_size + i;
				temp_y = __usad(y, half_kernel_size, j);//y - half_kernel_size + j;
				sum += image[temp_x*img_cols + temp_y]* kernel[i* kernel_size + j];
			}
		}*/

		// Manually unrolled loop
		int sum = 0;
		int temp_x;
		int temp_y;
		int half_kernel_size = kernel_size >> 2;
		sum += image[(x - 1) * img_cols + (y - 1)] * kernel[0];
		sum += image[(x - 1) * img_cols + (y)] * kernel[1];
		sum += image[(x - 1) * img_cols + (y + 1)] * kernel[2];

		sum += image[(x)*img_cols + (y - 1)] * kernel[3];
		sum += image[(x)*img_cols + (y)] * kernel[4];
		sum += image[(x)*img_cols + (y + 1)] * kernel[5];

		sum += image[(x + 1) * img_cols + (y - 1)] * kernel[6];
		sum += image[(x + 1) * img_cols + (y)] * kernel[7];
		sum += image[(x + 1) * img_cols + (y + 1)] * kernel[8];

		gpu::set(image_out, sum, x, y, img_rows, img_cols);
	}

	__syncthreads();
}


template <int kernel_idx>
__global__ void convolution_3(int* image, int img_rows, int img_cols, int* image_out, int img_row_from = 0) {
	/*
		N*N-pixels-per-thread implementation
	*/
	int x = (blockIdx.x * blockDim.x + threadIdx.x) * gpu::PER_THREAD_SIZE;
	int y = (blockIdx.y * blockDim.y + threadIdx.y) * gpu::PER_THREAD_SIZE;

	__shared__ int kernel_size;
	__shared__ float kernel[1024];

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		switch (kernel_idx)
		{
		case gpu::SOBEL_H:
			kernel_size = 3;
			memcpy(kernel, kernels[4], kernel_size * kernel_size * sizeof(float));
			break;
		case gpu::SOBEL_V:
			kernel_size = 3;
			memcpy(kernel, kernels[5], kernel_size * kernel_size * sizeof(float));
			break;
		case gpu::GAUSSIAN_KERNEL:
			kernel_size = GAUSSIAN_KERNEL_SIZE;
			memcpy(kernel, gaussian_kernel, kernel_size * kernel_size * sizeof(float));
			break;
		default:
			break;
		}
	}
	__syncthreads();
	int half_kernel_size = kernel_size >> 2;

	int temp_x;
	int temp_y;

	if (x < img_rows && y < img_cols) {
		for (int tx = 0; tx < gpu::PER_THREAD_SIZE; tx++) {
			for (int ty = 0; ty < gpu::PER_THREAD_SIZE; ty++) {
				int sum = 0;
				for (int i = 0; i < kernel_size; i++) {
					for (int j = 0; j < kernel_size; j++) {
						temp_x = x - half_kernel_size + i + tx;
						temp_y = y - half_kernel_size + j + ty;
						if (temp_x < img_rows && temp_y < img_cols) {
							sum += image[temp_x * img_cols + temp_y] * kernel[i * kernel_size + j];
						}
						else {
						}
						//sum = fabsf(sum);
						int x_ = x + tx;
						int y_ = y + ty;
						if (x_ < img_rows && y_ < img_cols)
							gpu::set(image_out, sum, x_, y_, img_rows, img_cols);
					}
				}

			}
		}
	}

	__syncthreads();
}

