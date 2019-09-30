﻿#pragma once
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
using namespace cv;
namespace cpu

{
	String img_path = "E:\\dev\\lF.jpg";

	const int H_KERNEL_3 = 1;
	const int H_KERNEL_5 = 2;
	const int V_KERNEL_3 = 3;
	const int V_KERNEL_5 = 4;
	const int GAUSSIAN_KERNEL_3 = 5;
	const int GAUSSIAN_KERNEL_5 = 6;
	const int SHARD_SIZE = 32;

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

	void imread_dense(cv::String, int** img, int mode,int *rows, int *cols, int ratio=1) {
		/*
		Read image to a dense array
		*/
		cv::Mat image = cv::imread(cpu::img_path, mode);
		*rows = image.rows / ratio;
		*cols = image.cols / ratio;
		cv::Mat image_resized(cv::Size(*cols, *rows), CV_32S);
		int img_size = sizeof(int) * (*rows) * (*cols);
		*img = (int*)malloc(img_size);
		cv::resize(image, image_resized, cv::Size(*cols, *rows), 0, 0);
		cpu::mat_to_dense(image_resized, *img);

	}

}


