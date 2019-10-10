
#define GAUSSIAN_KERNEL_SIZE 3


__device__ float* gaussian_kernel;
__device__ float* kernels[4];
__device__ int kernel_sizes[4];
const int gaussian_kernel_size = GAUSSIAN_KERNEL_SIZE;

__global__ void init_kernels() {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		gpu::get_kernel(gpu::H_KERNEL_3, &kernels[0], &kernel_sizes[0]);
		gpu::get_kernel(gpu::V_KERNEL_3, &kernels[1], &kernel_sizes[1]);
		gpu::get_kernel(gpu::DIAG_KERNEL_1, &kernels[2], &kernel_sizes[2]);
		gpu::get_kernel(gpu::DIAG_KERNEL_2, &kernels[3], &kernel_sizes[3]);
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
			res = max(gpu::get(img1, x, y, rows, cols), gpu::get(img2, x, y, rows, cols));
			gpu::set(dst, res, x, y, rows, cols);
		default:
			break;
		}
	}
	__syncthreads();

}

__global__ void convolution_0(int* image, int img_rows, int img_cols, int* kernel, int kernel_size, int* image_out) {
	/*
	Access image data using slow global memory
	*/
	//const int shard_size = blockDim.x;
	//const int shard_size_padded = blockDim.x + kernel_size/2;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sum = 0;


	if (x < 0 || x >= img_rows || y < 0 || y >= img_cols) {
		return;
	}
	else {

		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				int temp_x = x - kernel_size / 2 + i;
				int temp_y = y - kernel_size / 2 + j;
				// temp_x and temp_y are index corresponding to 
				// cells in convolution kernel
				if (temp_x < 0 || temp_x >= img_rows || temp_y < 0 || temp_y>img_cols) {
					sum += 0;
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

__global__ void convolution_1(int* image, int img_rows, int img_cols, int* image_out, int kernel_idx) {
	/*
	Kernel is generated at GPU side as
	Gaussian - Horizontal - Vertical pipeline
	*/

	int kernel_size;
	float kernel[1024];
	switch (kernel_idx)
	{
	case gpu::GAUSSIAN_KERNEL:
		kernel_size = GAUSSIAN_KERNEL_SIZE;
		memcpy(kernel, gaussian_kernel, kernel_size * kernel_size * sizeof(float));
		break;
	case gpu::H_KERNEL_3:
		kernel_size = 3;
		memcpy(kernel, kernels[0], kernel_size * kernel_size * sizeof(float));
		break;
	case gpu::V_KERNEL_3:
		kernel_size = 3;
		memcpy(kernel, kernels[1], kernel_size * kernel_size * sizeof(float));
		break;
	case gpu::DIAG_KERNEL_1:
		kernel_size = 3;
		memcpy(kernel, kernels[2], kernel_size * kernel_size * sizeof(float));
		break;
	case gpu::DIAG_KERNEL_2:
		kernel_size = 3;
		memcpy(kernel, kernels[3], kernel_size * kernel_size * sizeof(float));
		break;

	default:
		break;
	}

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sum = 0;

	if (x < 0 || x >= img_rows || y < 0 || y >= img_cols) {

	}

	else {

		for (int i = 0; i < kernel_size; i++) {
			for (int j = 0; j < kernel_size; j++) {
				int temp_x = x - kernel_size / 2 + i;
				int temp_y = y - kernel_size / 2 + j;
				// temp_x and temp_y are index corresponding to 
				// cells in convolution kernel
				if (temp_x < 0 || temp_x >= img_rows || temp_y < 0 || temp_y>img_cols) {
					sum += 0;
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