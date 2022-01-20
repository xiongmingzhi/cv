#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

 
int main(int argc, char** argv) {
 
	Mat src = imread("C:/Users/Administrator/Desktop/cv/lena.jpg"); // by default
	Mat dst, gray_src;
	if (src.empty())
	{
		printf("image load failed!\n");
		return -1;
	}
	namedWindow("input image");
	imshow("input image", src);
 
 
	GaussianBlur(src, dst, Size(3, 3), 0, 0);  //高斯模糊，使平滑
 
	//Sobel求x,y梯度
	Mat  grad_x, grad_y, grad_xy;
	//Sobel(gray_src, grad_x, CV_16S, 1, 0, 3); //经典Sobel算子
	//Sobel(gray_src, grad_y, CV_16S, 0, 1, 3);
	Scharr(gray_src, grad_x, CV_16S, 1, 0, 3); //改进Sobel算子
	Scharr(gray_src, grad_y, CV_16S, 0, 1, 3);
	convertScaleAbs(grad_x, grad_x); //可能为负，取绝对值，确保显示
	convertScaleAbs(grad_y, grad_y);
 
	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad_xy); //混合x,y梯度
 
	imshow("grad_xy", grad_xy);
 
	//imshow("output image", gray_src);
 
	waitKey(0);
	return 0;
 
}