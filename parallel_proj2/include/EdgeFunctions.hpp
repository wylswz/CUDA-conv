#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace cv;

// 函数声明
void SobelGradDirction(const Mat imageSource,Mat &imageSobelX,Mat &imageSobelY,double *&pointDrection);
void SobelAmplitude(const Mat imageGradX,const Mat imageGradY,Mat &SobelAmpXY);
void EdgeDetect(char path[256]);


void EdgeDetect(char path[256])
{
    Mat image;
    Mat imageCopy;  // 图片副本
    Mat imageGray;
    Mat imageGaussian;
    Mat imageEdge;
    
    image = imread(path);
//    imshow("original image", image);
    imageCopy = image;
    
    cvtColor(imageCopy, imageGray, COLOR_RGB2GRAY);
//    imshow("gray image, method 2", imageGray);
    
    GaussianBlur(imageGray, imageGaussian, Size(5, 5), 0, 0);
//    imshow("Gaussian blur", imageGauss);
    Mat imageSobelY;
    Mat imageSobelX;
    double *pointDirection=new double[(imageSobelX.cols-1)*(imageSobelX.rows-1)];  //定义梯度方向角数组
    SobelGradDirction(imageGaussian,imageSobelX,imageSobelY,pointDirection);  //计算X、Y方向梯度和方向角
//    imshow("Sobel Y",imageSobelY);
//    imshow("Sobel X",imageSobelX);
    Mat SobelGradAmpl;
    SobelAmplitude(imageSobelX,imageSobelY,SobelGradAmpl);   //计算X、Y方向梯度融合幅值
    imshow("Soble XYRange",SobelGradAmpl);
    
//    Canny(imageGauss, imageEdge, 100, 75);
//    imshow("edge detection", imageEdge);
    waitKey();

}

void SobelGradDirction(const Mat imageSource,Mat &imageSobelX,Mat &imageSobelY,double *&pointDrection)
{
    pointDrection=new double[(imageSource.rows-1)*(imageSource.cols-1)];
    for(int i=0;i<(imageSource.rows-1)*(imageSource.cols-1);i++)
    {
        pointDrection[i]=0;
    }
    imageSobelX=Mat::zeros(imageSource.size(),CV_32SC1);
    imageSobelY=Mat::zeros(imageSource.size(),CV_32SC1);
    uchar *P=imageSource.data;
    uchar *PX=imageSobelX.data;
    uchar *PY=imageSobelY.data;
 
    int step= (int)imageSource.step;
    int stepXY= (int)imageSobelX.step;
    int k=0;
    for(int i=1;i<(imageSource.rows-1);i++)
    {
        for(int j=1;j<(imageSource.cols-1);j++)
        {
            //通过指针遍历图像上每一个像素
            double gradY=P[(i-1)*step+j+1]+P[i*step+j+1]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[i*step+j-1]*2-P[(i+1)*step+j-1];
            PY[i*stepXY+j*(stepXY/step)]=abs(gradY);
            double gradX=P[(i+1)*step+j-1]+P[(i+1)*step+j]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1];
            PX[i*stepXY+j*(stepXY/step)]=abs(gradX);
            if(gradX==0)
            {
                gradX=0.00000000000000001;  //防止除数为0异常
            }
            pointDrection[k]=atan(gradY/gradX)*57.3;//弧度转换为度
            pointDrection[k]+=90;
            k++;
        }
    }
    convertScaleAbs(imageSobelX,imageSobelX);
    convertScaleAbs(imageSobelY,imageSobelY);
}

void SobelAmplitude(const Mat imageGradX,const Mat imageGradY,Mat &SobelAmpXY)
{
    SobelAmpXY=Mat::zeros(imageGradX.size(),CV_32FC1);
    for(int i=0;i<SobelAmpXY.rows;i++)
    {
        for(int j=0;j<SobelAmpXY.cols;j++)
        {
            SobelAmpXY.at<float>(i,j)=sqrt(imageGradX.at<uchar>(i,j)*imageGradX.at<uchar>(i,j)+imageGradY.at<uchar>(i,j)*imageGradY.at<uchar>(i,j));
        }
    }
    convertScaleAbs(SobelAmpXY,SobelAmpXY);
}
