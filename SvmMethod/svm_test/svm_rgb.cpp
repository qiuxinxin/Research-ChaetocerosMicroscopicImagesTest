//
//  svm2.cpp
//  svm_test
//
//  Created by qiuxinxin on 15/7/11.
//  Copyright (c) 2015年 qiu. All rights reserved.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;
typedef IplImage* IPL;

extern double compute_direction_angles(double,double,double,double,double *,double *,double *,double *);
extern IPL save_img(IPL,char *,string);


int main(int argc, char* argv[])
{
    IPL src1=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset2/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00.bmp",0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像,1为RGB图像
    Mat image2=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset2/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00.bmp");
    IPL src = cvCreateImage( cvSize( src1 -> width+2, src1 -> height+2 ), IPL_DEPTH_8U, src1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称

    int height=src->height, width=src->width;
    int i,j,m;
    
    Mat image=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Test/SvmMethod/Results-gsdam/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00_open.tif",0);
    
    cout<<image2.rows<<" "<<image2.cols<<endl;
    uchar *p,*p1;
    int k=0;
    Mat trainingData1(0,3,CV_32FC1);//正样本
    Mat trainingDataMat1;
    for( i = 0; i < image2.rows; i++)//原图image2比分割后图像行列小2个像素
    {
        p = image.ptr<uchar>(i+1);
        p1= image2.ptr<uchar>(i);
        for ( j = 0; j < image2.cols; j++)
        {
            if (float(p[j+1])==255)
            {
                trainingData1.push_back(float(p1[3*j]));
//                cout<<float(p1[3*(j+1)])<<endl;
                trainingData1.push_back(float(p1[3*j+1]));
                trainingData1.push_back(float(p1[3*j+2]));
                k=k+1;
            }
        }
    }
    cout<<k<<endl;
    trainingDataMat1=trainingData1.reshape(0,k);
    cout<<trainingDataMat1.size()<<endl;

//    int M=400;
//    Mat trainingData2(0,3,CV_32FC1);//负样本，取图像4个角10x10像素
//    Mat trainingDataMat2;
//    int ai[4]={0,0,height-12,height-12},aj[4]={0,width-12,0,width-12};
//    int bi[4]={10,10,height-2,height-2},bj[4]={10,width-2,10,width-2};
    int M=40000;
    Mat trainingData2(0,5,CV_32FC1);//负样本，取图像4个角100x100像素
    Mat trainingDataMat2;
    int ai[4]={0,0,height-102,height-102},aj[4]={0,width-102,0,width-102};
    int bi[4]={100,100,height-2,height-2},bj[4]={100,width-2,100,width-2};
    uchar *p3;
    for (m=0; m<4;m++)
    {
        for( i = ai[m]; i < bi[m]; i++)//原图image2比分割后图像小2个像素
        {
            p3 = image2.ptr<uchar>(i);
            for ( j = aj[m]; j < bj[m]; j++)
            {
                trainingData2.push_back(float(p3[3*j]));
                trainingData2.push_back(float(p3[3*j+1]));
                trainingData2.push_back(float(p3[3*j+2]));
            }
        }
    }
    trainingDataMat2=trainingData2.reshape(0,M);
    cout<<trainingDataMat2.size()<<endl;
    
    for (i=0;i<M;i++)//正负样本合并
    {
        trainingDataMat1.push_back(trainingDataMat2.row(i));
    }
    cout<<trainingDataMat1.size()<<endl;
    
    
    // Set up training data
    int labels[k+M];
    int val1=1,val2=-1;
    for (i=0;i<k;i++)
    {
        labels[i]=val1;
    }
    for (i=k;i<k+M;i++)
    {
        labels[i]=val2;
    }
    //    cout<<labels[100]<<endl;
    Mat labelsMat(k+M, 1, CV_32SC1, labels);
    //    cout<<labelsMat.size()<<endl;
    
    // Set up SVM's parameters
    Ptr<SVM> svm = SVM::create();
    svm->setC(0.1);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
    
    // Train the SVM
    svm->train(trainingDataMat1,ROW_SAMPLE,labelsMat);
    //
    uchar *p4;
    Mat predictData1(0,1,CV_32FC1);
    Mat predictDataMat;
    for( i = 0; i < image2.rows; i++)
    {
        p4=image2.ptr<uchar>(i);
        for ( j = 0; j < image2.cols; j++)
        {
            predictData1.push_back(float(p4[3*j]));
            predictData1.push_back(float(p4[3*j+1]));
            predictData1.push_back(float(p4[3*j+2]));
        }
    }
    predictDataMat=predictData1.reshape(0,image2.rows*image2.cols);
    cout<<predictDataMat.size()<<endl;
    Mat response;
    svm->predict(predictDataMat,response);
    cout<<response<<endl;
    Mat responseMat=response.reshape(0,image2.rows);
    
    Vec3b green(0,255,0), blue (255,0,0);
    for (i=0;i<image2.rows;i++)
    {
        for (j=0;j<image2.cols;j++)
        {
            if (responseMat.at<float>(i,j)== 1)
                image.at<Vec3b>(i,j)  = green;
            else if (responseMat.at<float>(i,j) == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
        
    }
    imwrite("/Users/qiuxinxin/temp/角毛藻显微图像/Test/SvmMethod/result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
    
}
