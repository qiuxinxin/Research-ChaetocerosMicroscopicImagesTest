//
//  other.cpp
//  GsdamTest
//
//  Created by qiuxinxin on 15/6/24.
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
extern IPL add_two_imgs2(IPL,IPL);
extern IPL extract_max_area_contour(IPL);
extern IPL and_template_original_imgs(IPL,IPL);
extern IPL save(IPL,char *,int);
extern IPL save_img(IPL,char *,string);
extern int check(char*);
extern int model(char*);



int main(int argc, char* argv[])
{
    IPL src1=cvLoadImage(argv[1],0);
    IPL src2=cvLoadImage(argv[2],0);

    IPL image_xz=cvCloneImage(src1);
    IPL image_yz=cvCloneImage(src2);

    IPL image_dilate1=cvCloneImage(image_xz);
    IPL image_dilate2=cvCloneImage(image_yz);

    IplConvKernel *element=0;
    IplConvKernel *element2=0;

    element=cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_ELLIPSE);//创建结构元素，定义了下面操作所需的模板，前两个参数定义模板大小，然后为参考点的坐标，模板的类型（椭圆）
    element2=cvCreateStructuringElementEx(2,2,0,0,CV_SHAPE_ELLIPSE);

    cvDilate(image_xz,image_dilate1,element2,5);//膨胀操作
    cvDilate(image_yz,image_dilate2,element2,5);
    string name9="_9_dilate1.tif";
    save_img(image_dilate1,argv[1],name9);
    string name10="_10_dilate2.tif";
    save_img(image_dilate2,argv[2],name10);
    
    cvReleaseImage(&image_dilate1);
    cvReleaseImage(&image_dilate2);
    
    return 0;
}
