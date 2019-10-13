#include <iostream>

#include <cmath>

using namespace cv;


void SobelGradDirction(const Mat imageSource,Mat &imageSobelX,Mat &imageSobelY,double *&pointDrection);
void SobelAmplitude(const Mat imageGradX,const Mat imageGradY,Mat &SobelAmpXY);
void EdgeDetect(char path[256]);


void EdgeDetect(char path[256])
{
    Mat image;
    Mat imageCopy;
    Mat imageGray;
    Mat imageGaussian;
    Mat imageEdge;
    
    image = imread(path);
//    imshow("original image", image);
    imageCopy = image;
    
	clock_t time_req;
	time_req = clock();

    cvtColor(imageCopy, imageGray, COLOR_RGB2GRAY);
//    imshow("gray image, method 2", imageGray);
    
    GaussianBlur(imageGray, imageGaussian, Size(3, 3), 0, 0);
//    imshow("Gaussian blur", imageGauss);
    Mat imageSobelY;
    Mat imageSobelX;
    double *pointDirection=new double[(imageSobelX.cols-1)*(imageSobelX.rows-1)];
    SobelGradDirction(imageGaussian,imageSobelX,imageSobelY,pointDirection);
//    imshow("Sobel Y",imageSobelY);
//    imshow("Sobel X",imageSobelX);
    Mat SobelGradAmpl;
    SobelAmplitude(imageSobelX,imageSobelY,SobelGradAmpl);
	time_req = clock() - time_req;
	std::cout << "Speed: " << (float)time_req / CLOCKS_PER_SEC << " seconds per image" << std::endl;
    //imshow("Soble XYRange",SobelGradAmpl);
    
//    Canny(imageGauss, imageEdge, 100, 75);
    //imshow("edge detection", imageEdge);
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
            double gradY=P[(i-1)*step+j+1]+P[i*step+j+1]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[i*step+j-1]*2-P[(i+1)*step+j-1];
            PY[i*stepXY+j*(stepXY/step)]=abs(gradY);
            double gradX=P[(i+1)*step+j-1]+P[(i+1)*step+j]*2+P[(i+1)*step+j+1]-P[(i-1)*step+j-1]-P[(i-1)*step+j]*2-P[(i-1)*step+j+1];
            PX[i*stepXY+j*(stepXY/step)]=abs(gradX);
            if(gradX==0)
            {
                gradX=0.00000000000000001;
            }
            pointDrection[k]=atan(gradY/gradX)*57.3;
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
