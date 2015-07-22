//
//  extract_maps2.cpp
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
    IPL image1=cvLoadImage(argv[1],0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像
    
    IPL image_border = cvCreateImage( cvSize( image1 -> width+2, image1 -> height+2 ), IPL_DEPTH_8U, image1->nChannels);
    
    IPL image_xz=cvCloneImage(image_border);
    IPL image_yz=cvCloneImage(image_border);
    
    int height=image_border->height, width=image_border->width;
    double **c1;
    double **c2;
    double **c3;
    double **c4;
    double **xmap;
    double **ymap;
    double **zmap;
    double **xzmap;
    double **yzmap;
    
    c1 = new double*[width];//x方向角的角度 new表示动态内存分配 c1作为一个指向指针的指针，它指向一个包含width个元素的指针数组
    c2 = new double*[width];
    c3 = new double*[width];
    c4 = new double*[width];
    xmap=new double*[width];
    ymap=new double*[width];
    zmap=new double*[width];
    xzmap=new double*[width];
    yzmap=new double*[width];
    
    
    cvCopyMakeBorder(image1, image_border, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
    
    for(int i=0; i<width; i++)
    {
        c1[i] = new double[height];//每个指针元素指向一个有height个元素的数组，这样就构建了有width行height列的数组
        c2[i] = new double[height];
        c3[i] = new double[height];
        c4[i] = new double[height];
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
        }
    }
    
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
        }
    }
    
    double xzmap_max=xzmap[0][0];
    double xzmap_min=xzmap[0][0];
    double yzmap_max=yzmap[0][0];
    double yzmap_min=yzmap[0][0];
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
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
            xzmap[x][y]=255*((xzmap[x][y]-xzmap_min)/(xzmap_max-xzmap_min));
            yzmap[x][y]=255*((yzmap[x][y]-yzmap_min)/(yzmap_max-yzmap_min));
            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
        }
    }
    
    string name1="_xz-map.tif";
    save_img(image_xz,argv[1],name1);
    
    string name2="_yz-map.tif";
    save_img(image_yz,argv[1],name2);
    
    
}

