//
//  main2.cpp
//  GsdamTest
//
//  Created by qiuxinxin on 15/6/23.
//  Copyright (c) 2015年 qiu. All rights reserved.
//
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <string.h>
#include <string>

using namespace std;
typedef IplImage* IPL;

extern double compute_direction_angles(double,double,double,double,double *,double *,double *,double *);
extern IPL map_binarization(IplImage*,IPL);
extern IPL and_two_binary_imgs(IPL,IPL);
extern IPL add_two_imgs(IPL,IPL);
extern IPL save_img(IPL,char *,string);



int main(int argc, char* argv[])
{
    IPL src1=cvLoadImage(argv[1],0);
    IPL src2=cvLoadImage(argv[2],0);
    IPL image_xz=cvCloneImage(src1);
    IPL image_yz=cvCloneImage(src2);

    
    IPL image_xz_and_yz=add_two_imgs(image_xz,image_yz);//与操作
    string name7="_1_xz_yz.tif";
    save_img(image_xz_and_yz,argv[1],name7);
    
    IplConvKernel *element=0;
    IPL image_close1=cvCloneImage(image_xz);
    IPL image_close2=cvCloneImage(image_yz);
    IPL image_temp1=cvCloneImage(image_xz);
    IPL image_temp2=cvCloneImage(image_yz);
    cvSetZero(image_temp1);
    cvSetZero(image_temp2);
    element=cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_ELLIPSE);
    cvMorphologyEx(image_xz,image_close1,image_temp1,element,CV_MOP_CLOSE,5);//闭运算
    cvMorphologyEx(image_yz,image_close2,image_temp2,element,CV_MOP_CLOSE,5);
    
    IPL image_xz_and_yz2=add_two_imgs(image_close1,image_close2);//与操作
    string name77="_2_xz_yz2.tif";
    save_img(image_xz_and_yz2,argv[1],name77);
    
    IPL imagesmoothM=cvCloneImage(image_xz_and_yz);
    IPL imagesmoothM2=cvCloneImage(image_xz_and_yz);

    cvSmooth(image_xz_and_yz,imagesmoothM,CV_MEDIAN,5,5);//中值滤波
    cvSmooth(imagesmoothM,imagesmoothM2,CV_MEDIAN,3,3);
    string name8="_4_smooth.tif";
    save_img(imagesmoothM,argv[1],name8);
    string name9="_5_smooth.tif";
    save_img(imagesmoothM2,argv[1],name9);

    cvReleaseImage(&image_xz_and_yz);
    
    return 0;
}