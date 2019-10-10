#pragma once
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <cstdint>

using namespace cv;
namespace cpu

{
	char * img_path = "E:\\dev\\Sheffield.jpg";
	char * img_path2 = "E:\\dev\\IMG_0788.jpg";

	const int H_KERNEL_3 = 1;
	const int H_KERNEL_5 = 2;
	const int V_KERNEL_3 = 3;
	const int V_KERNEL_5 = 4;
	const int GAUSSIAN_KERNEL_3 = 5;
	const int GAUSSIAN_KERNEL_5 = 6;
	const int GAUSSIAN_KERNEL = 7;
	const int DIAG_KERNEL_1 = 8;
	const int DIAG_KERNEL_2 = 9;
	const int SOBEL_H = 10;
	const int SOBEL_V = 11;

	const int SHARD_SIZE = 32;

	const int IMG_COMB_ADD = 101;
	const int IMG_COMB_MAX = 102;
	const int IMG_COMB_MIN = 103;
	const int IMG_COMB_MEAN = 104;
	const int IMG_COMB_MAGNITUDE = 105;
	
	template <typename T>
	T get(T* dense, int x, int y, int rows, int cols)
	{
		/*
		Indexing (x,y) element in a dense array

		*/
		int idx = x * cols + y;
		return dense[idx];
	}

	template <typename T>
	void set(T* dense, T value, int x, int y, int rows, int cols)
	{
		/*
		Setting (x, y) element in a dense array
		*/
		int idx = x * cols + y;
		dense[idx] = value;
	}

	template <typename T>
	void sparse2dense(T* dense, T** sparse, int rows, int cols)
	{

	}

	template <typename T>
	void dense2sparse(T* dense, T** sparse, int rows, int cols)
	{

	}

	void get_kernel(int type, int** kernel, int* kernel_size) {
		switch (type)
		{
		case H_KERNEL_3:
			*kernel_size = 3;
			int size = 3 * 3 * sizeof(int);
			*kernel = (int*)malloc(size);
			(*kernel)[0] = 1; (*kernel)[1] = 1; (*kernel)[2] = 1;
			(*kernel)[3] = 0; (*kernel)[4] = 0; (*kernel)[5] = 0;
			(*kernel)[6] = -1; (*kernel)[7] = -1; (*kernel)[8] = -1;

			break;
		}
	}

	void mat_to_dense(cv::Mat matrix, int* dense);

	void cv_util_test() {

		Mat img = imread(img_path, IMREAD_GRAYSCALE);
		if (!img.data) {
			
			std::cout<<"Image does not exist"<<std::endl;
			return;	
		}

		imshow("myimg", img);
		waitKey(0);
		
	}


	void mat_to_dense(cv::Mat mat, int* dense) {
		for (int i = 0; i < mat.rows; i++) {
			for (int j = 0; j < mat.cols; j++) {
				int elem = cpu::get(mat.data, i, j, mat.rows, mat.cols);
				cpu::set(dense, elem, i, j, mat.rows, mat.cols);
			}
		}
	}

	void imread_dense(cv::String img_path, int** img, int mode,int *rows, int *cols, int ratio=1) {
		/*
		Read image to a dense array
		*/
		cv::Mat image = cv::imread(img_path, mode);
		*rows = image.rows / ratio;
		*cols = image.cols / ratio;
		cv::Mat image_resized(cv::Size(*cols, *rows), CV_32S);
		int img_size = sizeof(int) * (*rows) * (*cols);
		*img = (int*)malloc(img_size);
		cv::resize(image, image_resized, cv::Size(*cols, *rows), 0, 0);
		cpu::mat_to_dense(image_resized, *img);

	}

	void imread_dense(cv::String img_path, int** img, int mode, int rows, int cols) {
		/*
		Read image given width and height
		*/
		cv::Mat image = cv::imread(img_path, mode);
		cv::Mat image_resized(cv::Size(cols, rows), CV_32S);
		int img_size = sizeof(int) * rows * cols;
		*img = (int*)malloc(img_size);
		cv::resize(image, image_resized, cv::Size(cols, rows), 0, 0);
		cpu::mat_to_dense(image_resized, *img);
	}

	void imshow(int* img, int rows, int cols,std::string window_name="imshow", int window_x = 500, int window_y = 500) {
		Mat img_to_show(cv::Size(cols, rows), CV_32S, img);
		img_to_show.convertTo(img_to_show, CV_32F, 1 / 255.0);

		cv::namedWindow(window_name, 0);
		cv::resizeWindow(window_name, window_x, window_y);
		cv::imshow(window_name, img_to_show);
		waitKey(0);
	}

}


