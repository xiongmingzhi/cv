#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
int main()
{
    //String pbtxt = "C:/Users/Administrator/Desktop/cv/tf_BP_regress.pbtxt";
    String pb = "C:/Users/Administrator/Desktop/cv/tf_BP_regress.pb";
    dnn::Net net = cv::dnn::readNetFromTensorflow(pb);

    // float wrinkle[1][6]={{1.568,3.529,3.499,0.576,0.509,0}};
    float wrinkle[1][6]={{7.729,9.118,6.368,5.137,5.02,2.044}};

    // Mat input =(Mat <CV_32F1>(1,7)  <<1.568,3.529,3.499,0.576,0.509,0,9.681  );
    Mat input =Mat(1,6,CV_32FC1,wrinkle) ;
    net.setInput(input);

    Mat tmp = net.forward();
    cout<<"tmp"<<tmp<<endl;
    float result=tmp.at<float>(0,0);

    cout<< result<<endl;
    return 0;

}