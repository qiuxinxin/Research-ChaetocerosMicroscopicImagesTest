//
//  svm_maps_resize.cpp
//  svm_test
//
//  Created by qiuxinxin on 15/7/12.
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


int main(int argc, char* argv[])
{
    
    //    IPL image1=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00.bmp",0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像
    //    Mat image2=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00.bmp");
    
    IPL image1=cvLoadImage(argv[1],0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像
    Mat image2=imread(argv[1]);
    
    double scale=0.15;
    CvSize czsize;
    czsize.width=image1->width*scale;
    czsize.height=image1->height*scale;
    IPL image1_resize=cvCreateImage(czsize, image1->depth, image1->nChannels);
    cvResize(image1, image1_resize);
    cout<<image1_resize->width<<" "<<image1_resize->height<<" "<<image1->nChannels<<endl;
    //    cvShowImage("1",image1_resize);
    //    waitKey(0);
    
    Size img2_dsize = Size(image2.cols*scale,image2.rows*scale);
    Mat img_ori_re = Mat(img2_dsize,CV_8UC3);
    resize(image2,img_ori_re,img2_dsize);
    cout<<img_ori_re.size()<<" "<<endl;
    //    imshow("1",img_ori_re);
    //    waitKey(0);
    
    IPL image_border = cvCreateImage( cvSize( image1_resize -> width+2, image1_resize -> height+2 ), IPL_DEPTH_8U, image1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称
    
    int height=image_border->height, width=image_border->width;
    printf("h=%d w=%d\n",height,width);
    double **xmap;
    double **ymap;
    double **zmap;
    double **xzmap;
    double **yzmap;
    
    xmap=new double*[width];
    ymap=new double*[width];
    zmap=new double*[width];
    xzmap=new double*[width];
    yzmap=new double*[width];
    
    
    cvCopyMakeBorder(image1_resize, image_border, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
    //        cvShowImage("1",image_border);
    //        waitKey(0);
    
    for(int i=0; i<width; i++)
    {
        xmap[i]=new double[height];
        ymap[i]=new double[height];
        zmap[i]=new double[height];
        xzmap[i]=new double[height];
        yzmap[i]=new double[height];
    }
    
    int x,y;
    CvScalar Ix0y0,Ix0y1,Ix1y1,Ix1y0;//cvscalar的结构为一个double型的4成员的数组，它提供了四个成员，最多就可对应四个通道
    double w1,w2,w3,* w1_z,* w2_z,* w3_z,dis,* dis_z;
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            Ix0y0=cvGet2D(image_border,y,x);//提取图像(y,x)的像素值，与通常坐标系相反
            Ix0y1=cvGet2D(image_border,(y+1),x);
            Ix1y1=cvGet2D(image_border,(y+1),(x+1));
            Ix1y0=cvGet2D(image_border,y,(x+1));
            w1_z=&w1;
            w2_z=&w2;
            w3_z=&w3;
            dis_z=&dis;
            compute_direction_angles(Ix0y0.val[0],Ix0y1.val[0],Ix1y0.val[0],Ix1y1.val[0],w1_z,w2_z,w3_z,dis_z);
            //此函数作用是分别求出xyz方向的三个夹角大小
            xmap[x][y]=w1;//w1是法向量与与x轴的夹角；
            ymap[x][y]=w2;
            zmap[x][y]=w3;
            //            cout<<ymap[x][y]<<endl;
        }
    }
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
            //                        cout<<yzmap[x][y]<<endl;
        }
    }
    
    
    int i,j,m;
    
    //    Mat img_seg=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Test/SvmMethod/Results-gsdam-new/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00_open.tif",0);
    Mat img_seg=imread(argv[2],0);
    
    Size img_dsize = Size(img_seg.cols*scale,img_seg.rows*scale);
    Mat img_seg_re = Mat(img_dsize,CV_8UC1);//image_seg.depth()
    resize(img_seg,img_seg_re,img_dsize);
    //    imshow("1",img_seg_re);
    //    waitKey(0);
    
    cout<<img_ori_re.rows<<" "<<img_ori_re.cols<<endl;
    int column=5;
    int k=0;
    Mat trainingData1(0,1,CV_32FC1);//正样本
    Mat trainingDataMat1;
    for( i = 0; i < img_ori_re.rows; i++)
    {
        for (int j=0; j<img_ori_re.cols; j++)
        {
            if (img_seg_re.at<uchar>(i, j)==255)
            {
                trainingData1.push_back(float(xmap[j+1][i+1]));
                trainingData1.push_back(float(ymap[j+1][i+1]));
                trainingData1.push_back(float(zmap[j+1][i+1]));
                trainingData1.push_back(float(xzmap[j+1][i+1]));
                trainingData1.push_back(float(yzmap[j+1][i+1]));
                k=k+1;
            }
        }
    }
    //    cout<<trainingData1<<endl;
    //    cout<<k<<endl;
    trainingDataMat1=trainingData1.reshape(0,k);
    cout<<trainingDataMat1.size()<<endl;
    
    int N=sqrt(k*0.6/4);//根据正样本的多少决定负样本数目
    int M=N*N*4;
    Mat trainingData2(0,1,CV_32FC1);//负样本，取图像4个角10x10像素
    Mat trainingDataMat2;
    int ai[4]={0,0,img_ori_re.rows-N,img_ori_re.rows-N},aj[4]={0,img_ori_re.cols-N,0,img_ori_re.cols-N};
    int bi[4]={N,N,img_ori_re.rows,img_ori_re.rows},bj[4]={N,img_ori_re.cols,N,img_ori_re.cols};
    //    int M=40000;
    //    Mat trainingData2(0,5,CV_32FC1);//负样本，取图像4个角100x100像素
    //    Mat trainingDataMat2;
    //    int ai[4]={0,0,img_ori_re.rows-100,img_ori_re.rows-100},aj[4]={0,img_ori_re.cols-100,0,img_ori_re.cols-100};
    //    int bi[4]={100,100,img_ori_re.rows,img_ori_re.rows},bj[4]={100,img_ori_re.cols,100,img_ori_re.cols};
    
    for (m=0; m<4;m++)
    {
        for( i = ai[m]; i < bi[m]; i++)//原图image2比分割后图像小2个像素
        {
            for ( j = aj[m]; j < bj[m]; j++)
            {
                trainingData2.push_back(float(xmap[j+1][i+1]));
                trainingData2.push_back(float(ymap[j+1][i+1]));
                trainingData2.push_back(float(zmap[j+1][i+1]));
                trainingData2.push_back(float(xzmap[j+1][i+1]));
                trainingData2.push_back(float(yzmap[j+1][i+1]));
            }
        }
    }
    trainingDataMat2=trainingData2.reshape(0,M);
    cout<<trainingDataMat2.size()<<endl;
    //    cout<<trainingDataMat2<<endl;
    
    for (i=0;i<M;i++)//正负样本合并
    {
        trainingDataMat1.push_back(trainingDataMat2.row(i));
    }
    cout<<trainingDataMat1.size()<<endl;
    
    //特征向量归一化
    Mat trainingDataMat=Mat(k+M, column, CV_32FC1);
    
    for (i=0; i<column; i++)
    {
        double min=1, max=1;
        Mat temp=trainingDataMat1.colRange(i,i+1).clone();//提取trainingDataMat1的每一列特征
        //        cout<<temp<<endl;
        minMaxIdx(temp,&min,&max);
        //        cout<<min<<" "<<max<<endl;
        for (j=0;j<trainingDataMat1.rows;j++)
        {
            trainingDataMat.at<float>(j,i)=(trainingDataMat1.at<float>(j,i)-min)/(max-min);
            //            cout<<trainingDataMat.at<float>(j,i)<<" "<<trainingDataMat1.at<float>(j,i)<<endl;
        }
    }
    
    
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
    svm->train(trainingDataMat,ROW_SAMPLE,labelsMat);
    
    Mat predictData1(0,1,CV_32FC1);
    Mat predictDataMat1;
    for( i = 0; i < img_ori_re.rows; i++)
    {
        for ( j = 0; j < img_ori_re.cols; j++)
        {
            predictData1.push_back(float(xmap[j+1][i+1]));
            predictData1.push_back(float(ymap[j+1][i+1]));
            predictData1.push_back(float(zmap[j+1][i+1]));
            predictData1.push_back(float(xzmap[j+1][i+1]));
            predictData1.push_back(float(yzmap[j+1][i+1]));
        }
    }
    predictDataMat1=predictData1.reshape(0,img_ori_re.rows*img_ori_re.cols);
    //    cout<<predictDataMat.size()<<endl;
    Mat predictDataMat=Mat(img_ori_re.rows*img_ori_re.cols, column, CV_32FC1);
    for (i=0; i<column; i++)
    {
        double min=1, max=1;
        Mat temp=predictDataMat1.colRange(i,i+1).clone();
        minMaxIdx(temp,&min,&max);
        for (j=0;j<predictDataMat1.rows;j++)
        {
            predictDataMat.at<float>(j,i)=(predictDataMat1.at<float>(j,i)-min)/(max-min);
        }
    }
    
    Mat response;
    svm->predict(predictDataMat,response);
    cout<<response<<endl;
    Mat responseMat=response.reshape(0,img_ori_re.rows);
    
    Vec3b green(0,255,0), blue (255,0,0);
    for (i=0;i<img_ori_re.rows;i++)
    {
        for (j=0;j<img_ori_re.cols;j++)
        {
            if (responseMat.at<float>(i,j)== 1)
                img_ori_re.at<Vec3b>(i,j)  = green;
            else if (responseMat.at<float>(i,j) == -1)
                img_ori_re.at<Vec3b>(i,j)  = blue;
        }
        
    }
    string savingfile(argv[1]);
    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
    string add="_svm-5maps.tif";
    shortstring+=add;//加入新的标示及扩展名
    imwrite(shortstring,img_ori_re);
    
    //    imwrite("/Users/qiuxinxin/temp/角毛藻显微图像/TestOpencv/result-nonormal.png", img_ori_re);
    //    imshow("SVM Simple Example", img_ori_re); // show it to the user
    //    waitKey(0);
    
    for(int i=0; i<width; i++)
    {
        delete []xmap[i];
        delete []ymap[i];
        delete []zmap[i];
        delete []xzmap[i];
        delete []yzmap[i];
    }
    
    delete []xmap;
    delete []ymap;
    delete []zmap;
    delete []xzmap;
    delete []yzmap;
    
    return 0;
}

