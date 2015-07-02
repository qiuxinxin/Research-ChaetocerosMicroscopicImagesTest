//
//  add_two_imgs2.cpp
//  GsdamTest
//
//  Created by qiuxinxin on 15/6/23.
//  Copyright (c) 2015年 qiu. All rights reserved.
//

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;
typedef IplImage* IPL;

IPL add_two_imgs(IPL src1,IPL src2)
{
    IPL bimage=cvCloneImage(src1);
    cvZero(bimage);
    int height=src1->height,width=src1->width;
    double src1temp,src2temp;
    
    for(int x=0; x<height; x++)
    {
        for(int y=0; y<width; y++)
        {
            src1temp=cvGet2D(src1,x,y).val[0];
            src2temp=cvGet2D(src2,x,y).val[0];
            if (src1temp==src2temp)
            {
                cvSet2D(bimage,x,y,cvRealScalar(src2temp));
            }
            else
            {
                cvSet2D(bimage,x,y,cvRealScalar(255));
            }
        }
    }
    return bimage;
}

