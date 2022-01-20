#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

vector<string> split(const string& str, const string& delim) {
    vector<string> res;
    if ("" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char *strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while (p) {
        string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }

    return res;
}

void vector2Mat(vector< vector<float> > src,Mat & dst,int type)
{
    Mat temp(src.size(),src.at(0).size(),type);
    for(int i=0; i<temp.rows; ++i)
        for(int j=0; j<temp.cols; ++j)
            temp.at<float>(i, j) = src.at(i).at(j);
    temp.copyTo(dst);

}

vector<vector<float>> read_csv(vector<vector<float>> &swp)
{
    ifstream inFile("C:\\Users\\Administrator\\Desktop\\cv\\1273.csv", ios::in);
    if (!inFile)
    {
        cout << "打开文件失败！" << endl;
        exit(1);
    }

    string line;
    string field;
    string str;

    int count =0;

    while (getline(inFile, line))//getline(inFile, line)表示按行读取CSV文件中的数据
    {
        vector<string> field;
        string text;
        istringstream sin(line); //将整行字符串line读入到字符串流sin中
        field = split(line,",");
        field.size();
        vector<float> wrikle2;
        for(int i = 0;i < field.size();i++)
        {
            float aaa = atof(field[i].c_str());
            wrikle2.push_back(aaa);

        }
        swp.push_back(wrikle2);
        count++;
    }
    //测试代码
//    cout << swp[5][1] <<endl;
//    cout << count <<endl;
    inFile.close();
    return swp;
}

int main()
{
    String pb = "C:/Users/Administrator/Desktop/cv/model/model.pb";
    String prototxt = "C:/Users/Administrator/Desktop/cv/model/lenet2.pbtxt";
    dnn::Net net = cv::dnn::readNetFromTensorflow(pb, prototxt);

    vector<vector<float>> wrinkle ;
    read_csv(wrinkle);

    Mat input;
    vector2Mat(wrinkle,input,CV_32FC1);
    net.setInput(input);


    Mat tmp = net.forward();
    cout<<"tmp"<<tmp<<endl;
    float result=tmp.at<float>(0,0);

    cout<< result<<endl;
    return 0;
}