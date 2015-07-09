//
//  svmtest.cpp
//  cluster_test
//
//  Created by qiuxinxin on 15/7/8.
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
    IPL src1=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset/北方角毛藻/北方角毛藻_环面观_南海_00000000_115拖网_040_00.bmp",0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像,1为RGB图像
    Mat image2=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset/北方角毛藻/北方角毛藻_环面观_南海_00000000_115拖网_040_00.bmp");
    IPL src = cvCreateImage( cvSize( src1 -> width+2, src1 -> height+2 ), IPL_DEPTH_8U, src1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称

    IPL image_x=cvCloneImage(src);
    IPL image_y=cvCloneImage(src);
    IPL image_z=cvCloneImage(src);
    IPL image_xy=cvCloneImage(src);
    IPL image_xz=cvCloneImage(src);
    IPL image_yz=cvCloneImage(src);
    
    int height=src->height, width=src->width;
//            printf("h=%d w=%d\n",height,width);
    double **c1;
    double **c2;
    double **c3;
    double **c4;
    double **xmap;
    double **ymap;
    double **zmap;
    double **xymap;
    double **xzmap;
    double **yzmap;
    
    c1 = new double*[width];//x方向角的角度 new表示动态内存分配 c1作为一个指向指针的指针，它指向一个包含width个元素的指针数组
    c2 = new double*[width];
    c3 = new double*[width];
    c4 = new double*[width];
    xmap=new double*[width];
    ymap=new double*[width];
    zmap=new double*[width];
    xymap=new double*[width];
    xzmap=new double*[width];
    yzmap=new double*[width];
    
    
    cvCopyMakeBorder(src1, src, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
//    cvCopyMakeBorder(src2, srcc, cvPoint(1,1), IPL_BORDER_CONSTANT);
    
    for(int i=0; i<width; i++)
    {
        c1[i] = new double[height];//每个指针元素指向一个有height个元素的数组，这样就构建了有width行height列的数组
        c2[i] = new double[height];
        c3[i] = new double[height];
        c4[i] = new double[height];
        xmap[i]=new double[height];
        ymap[i]=new double[height];
        zmap[i]=new double[height];
        xymap[i]=new double[height];
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
            Ix0y0=cvGet2D(src,y,x);//提取图像(y,x)的像素值，与通常坐标系相反
            //            printf("y=%d x=%d ixoyo=%f\n",y,x,Ix0y0.val[0]);
            Ix0y1=cvGet2D(src,(y+1),x);
            Ix1y1=cvGet2D(src,(y+1),(x+1));
            //            printf("y=%d x=%d ixoyo=%f\n",y+1,x+1,Ix1y1.val[0]);
            Ix1y0=cvGet2D(src,y,(x+1));
            w1_z=&w1;
            w2_z=&w2;
            w3_z=&w3;
            dis_z=&dis;
            compute_direction_angles(Ix0y0.val[0],Ix0y1.val[0],Ix1y0.val[0],Ix1y1.val[0],w1_z,w2_z,w3_z,dis_z);
            //此函数作用是分别求出xyz方向的三个夹角大小
            c1[x][y]=w1;//w1是法向量与与x轴的夹角；
            c2[x][y]=w2;
            c3[x][y]=w3;
        }
    }
    
    double xmax,xmin,ymax,ymin,zmax,zmin;
    double c1yx,c2yx,c3yx;
    xmax=c1[0][0];
    xmin=c1[0][0];
    ymax=c2[0][0];
    ymin=c2[0][0];
    zmax=c3[0][0];
    zmin=c3[0][0];
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            c1yx=c1[x][y];
            if (c1yx>xmax)
                xmax=c1yx;//max(theta_x)
            else {};
            if (c1yx<xmin)
                xmin=c1yx;//min(theta_x)
            else {};
            
            c2yx=c2[x][y];
            if (c2yx>ymax)
                ymax=c2yx;
            else {};
            if (c2yx<ymin)
                ymin=c2yx;
            else {};
            
            c3yx=c3[x][y];
            if (c3yx>zmax)
                zmax=c3yx;
            else {};
            if (c3yx<zmin)
                zmin=c3yx;
            else {};
        }
    }
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xmap[x][y]=255*(c1[x][y]-xmin)/(xmax-xmin);
            ymap[x][y]=255*(c2[x][y]-ymin)/(ymax-ymin);
            zmap[x][y]=255*(c3[x][y]-zmin)/(zmax-zmin);
//                        printf("x=%f\n",xmap[x][y]);
            cvSet2D(image_x,y,x,cvRealScalar(xmap[x][y]));//将map的值赋给图像中每个像素点
            cvSet2D(image_y,y,x,cvRealScalar(ymap[x][y]));//cvrealscalar把数组的第一个值置为想要的数（即为 double val0 ），其他三个数均为零。
            cvSet2D(image_z,y,x,cvRealScalar(zmap[x][y]));
            
        }
    }
    
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xymap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+ymap[x][y]*ymap[x][y]);
            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
            cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
        }
    }
    //
    double xymap_max=xymap[0][0];
    double xymap_min=xymap[0][0];
    double xzmap_max=xzmap[0][0];
    double xzmap_min=xzmap[0][0];
    double yzmap_max=yzmap[0][0];
    double yzmap_min=yzmap[0][0];
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            if (xymap[x][y]>xymap_max)
            {
                xymap_max=xymap[x][y];
            }
            else {}
            if (xymap[x][y]<xymap_min)
            {
                xymap_min=xymap[x][y];
            }
            else {}
            if (xzmap[x][y]>xzmap_max)
            {
                xzmap_max=xzmap[x][y];
            }
            else {}
            if (xzmap[x][y]<xzmap_min)
            {
                xzmap_min=xzmap[x][y];
            }
            else {}
            if (yzmap[x][y]>yzmap_max)
            {
                yzmap_max=yzmap[x][y];
            }
            else {}
            if (yzmap[x][y]<yzmap_min)
            {
                yzmap_min=yzmap[x][y];
            }
            else {}
        }
    }
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xymap[x][y]=255*( (xymap[x][y]-xymap_min)/(xymap_max-xymap_min));
            //               printf("xy=%f\n",xymap[x][y]);
            xzmap[x][y]=255*( (xzmap[x][y]-xzmap_min)/(xzmap_max-xzmap_min));//对xzmap,yzmap进行255归一化
            yzmap[x][y]=255*( (yzmap[x][y]-yzmap_min)/(yzmap_max-yzmap_min));
            //                printf("y=%d x=%d xxx=%f yyy=%f zzz=%f xxy=%f xxz=%f xyz=%f\n",y,x,xmap[x][y],ymap[x][y],zmap[x][y],xymap[x][y],xzmap[x][y],yzmap[x][y]);
            cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
        }
    }
//    cout<<ymap[1974][0]<<endl;
    for(int i=0; i<width; i++)
    {
        delete []c1[i];//先撤销指针元素所指向的数组
        delete []c2[i];
        delete []c3[i];
        delete []c4[i];
//        delete []xmap[i];
//        delete []ymap[i];
//        delete []zmap[i];
//        delete []xymap[i];
//        delete []xzmap[i];
//        delete []yzmap[i];
    }
    delete []c1;//在销毁的过程，先销毁指针数组每个元素指向的数组，然后再销毁这个指针数组
    delete []c2;
    delete []c3;
    delete []c4;
//    delete []xmap;
//    delete []ymap;
//    delete []zmap;
//    delete []xymap;
//    delete []xzmap;
//    delete []yzmap;
    
    int i,j;
    
//    Mat imagex=cvarrToMat(image_x);//iplimage转mat
//    Mat imagey=cvarrToMat(image_y);
//    Mat imagez=cvarrToMat(image_z);
//    Mat imagexz=cvarrToMat(image_xz);
//    Mat imageyz=cvarrToMat(image_yz);
//    //cout<<imageyz<<endl;
//    Mat image1=cvarrToMat(src);
////    Mat image2=cvarrToMat(srcc);
//    Mat image3=cvarrToMat(src2);

    Mat image=imread("/Users/qiuxinxin/temp/角毛藻显微图像/cluster/cluster_test/Results-gsdam/北方角毛藻/北方角毛藻_环面观_南海_00000000_115拖网_040_00_open.tif");
    
//    cout<<image<<endl;
//    cout<<image.rows<<" "<<image.cols<<endl;
    uchar *p;
    int k=0;

//    float trainingData1[10][5];
    Mat trainingData1(0,5,CV_32FC1);
    Mat trainingDataMat;
    for( i = 0; i < image.rows; i++)
    {
        p = image.ptr<uchar>(i);
//        cout<<p<<endl;
        for ( j = 0; j < image.cols; j++)
        {
//            cout<<float(p[j+1])<<endl;
            if (float(p[j])==255)
            {
                trainingData1.push_back(float(xmap[j][i]));
                trainingData1.push_back(float(ymap[j][i]));
                trainingData1.push_back(float(zmap[j][i]));
                trainingData1.push_back(float(xzmap[j][i]));
                trainingData1.push_back(float(yzmap[j][i]));
                k=k+1;
//                cout<<float(xmap[j][i])<<" "<<float(ymap[j][i])<<" "<<zmap[j][i]<<" "<<float(xzmap[j][i])<<" "<<float(yzmap[j][i])<<endl;
            }
        }
    }
    trainingDataMat=trainingData1.reshape(0,k);
//cout<<trainingData.rows<<endl;
    
    // Set up training data
    int labels[k];
    int val=1;
    for (i=0;i<k;i++)
    {
        labels[i]=val;
    }
//    cout<<labels[100]<<endl;
    Mat labelsMat(k, 1, CV_32SC1, labels);

    // Set up SVM's parameters
    Ptr<SVM> svm = SVM::create();
    svm->setC(0.1);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));

    // Train the SVM
    svm->train(trainingDataMat,ROW_SAMPLE,labelsMat);
    
    Mat predictData1(0,5,CV_32FC1);
    Mat predictDataMat;
    for( i = 0; i < image2.rows; i++)
    {
        p = image.ptr<uchar>(i);
        for ( j = 0; j < image2.cols; j++)
        {
            predictData1.push_back(float(xmap[j][i]));
            predictData1.push_back(float(ymap[j][i]));
            predictData1.push_back(float(zmap[j][i]));
            predictData1.push_back(float(xzmap[j][i]));
            predictData1.push_back(float(yzmap[j][i]));
        }
    }
    predictDataMat=predictData1.reshape(0,image2.rows*image2.cols);
    
    float response = svm->predict(predictDataMat);

//    Vec3b green(0,255,0), blue (255,0,0);
//    // Show the decision regions given by the SVM
//    for (int i = 0; i < image.rows; ++i)
//        for (int j = 0; j < image.cols; ++j)
//        {
//            Mat sampleMat = (Mat_<float>(1,2) << i,j);
//            cout<<(Mat_<float>(1,2) << i,j)<<endl;
//            float response = svm->predict(sampleMat);
//            
//            if (response == 1)
//                image.at<Vec3b>(i,j)  = green;
//            else //if (response == -1)
//                image.at<Vec3b>(i,j)  = blue;
//        }
//
//    // Show the training data
//    int thickness = -1;
//    int lineType = 8;
//    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType );
//    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType );
//    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType );
//    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType );
//    
//    // Show support vectors
//    thickness = 2;
//    lineType  = 8;
//    Mat sv = svm->getSupportVectors();
//    
//    for (int i = 0; i < sv.rows; ++i)
//    {
//        const float* v = sv.ptr<float>(i);
//        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
//    }
//    
////    imwrite("result.png", image);        // save the image
//    
    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
    
}