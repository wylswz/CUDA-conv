__device__ float* gaussian_kernel;
__device__ int gaussian_kernel_size=1;

__device__ int* kernels[2];
__device__ int kernel_sizes[2];


__global__ void init_kernels() {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		gpu::get_kernel(gpu::H_KERNEL_3, &kernels[0], &kernel_sizes[0]);
		gpu::get_kernel(gpu::V_KERNEL_3, &kernels[1], &kernel_sizes[1]);
		gpu::get_gaussian_kernel(&gaussian_kernel, gaussian_kernel_size);
		gpu::normalize(gaussian_kernel, gaussian_kernel_size * gaussian_kernel_size);
		printf("%f", gaussian_kernel[0]);
	}
	__syncthreads();
}

__global__ void init_gaussian_kernel() {
	
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
		gpu::abs(&sum);
		//__syncthreads();
		gpu::set(image_out, sum, x, y, img_rows, img_cols);
	}
	__syncthreads();
}

__global__ void convolution_1(int* image, int img_rows, int img_cols, int* image_out) {
	/*
	Kernel is generated at GPU side as
	Gaussian - Horizontal - Vertical pipeline
	*/

	//int* kernel = kernels[1];
	//int kernel_size = kernel_sizes[1];

	float* kernel = gaussian_kernel;
	int kernel_size = gaussian_kernel_size;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sum = 0;
	//gpu::get_kernel(gpu::H_KERNEL_3, &kernel, &kernel_size);
	//__syncthreads();

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

		gpu::abs(&sum);
		//__syncthreads();
		gpu::set(image_out, sum, x, y, img_rows, img_cols);

	}
	__syncthreads();
}