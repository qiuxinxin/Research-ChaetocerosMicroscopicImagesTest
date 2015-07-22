//
//  main.cpp
//  region_test
//
//  Created by qiuxinxin on 15/7/21.
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
    
    //    IPL image1=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00.bmp",0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像
    //    Mat image2=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Dataset/丹麦角毛藻/丹麦角毛藻_宽环面观_青岛沿海_20041210_00_040_00.bmp");
    
    IPL image1=cvLoadImage(argv[1],0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像
    
//    double scale=0.15;
//    CvSize czsize;
//    czsize.width=image1->width*scale;
//    czsize.height=image1->height*scale;
//    IPL image1_resize=cvCreateImage(czsize, image1->depth, image1->nChannels);
//    cvResize(image1, image1_resize);
////    cout<<image1_resize->width<<" "<<image1_resize->height<<" "<<image1->nChannels<<endl;
//    
//    IPL image_border = cvCreateImage( cvSize( image1_resize -> width+2, image1_resize -> height+2 ), IPL_DEPTH_8U, image1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称
    IPL image_border = cvCreateImage( cvSize( image1 -> width+2, image1 -> height+2 ), IPL_DEPTH_8U, image1->nChannels);
    
    IPL image_xz=cvCloneImage(image_border);
    IPL image_yz=cvCloneImage(image_border);
    
    
    int height=image_border->height, width=image_border->width;
//    printf("h=%d w=%d\n",height,width);
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
    
    
    cvCopyMakeBorder(image1, image_border, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
    
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
        }
    }
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
        }
    }
   
    string name1="_xz-map.tif";
    save_img(image_xz,argv[1],name1);
    
    string name2="_yz-map.tif";
    save_img(image_yz,argv[1],name2);
    
    
}
