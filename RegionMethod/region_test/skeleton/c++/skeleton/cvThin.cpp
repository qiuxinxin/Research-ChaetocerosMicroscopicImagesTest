//
//  cvThin.cpp
//  skeleton
//
//  Created by qiuxinxin on 15/8/6.
//  Copyright (c) 2015年 qiu. All rights reserved.
//zhang-sue
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

void cvThin(cv::Mat& src, cv::Mat& dst, int intera)
{
    if(src.type()!=CV_8UC1)
    {
        printf("只能处理二值或灰度图像\n");
        return;
    }
    //非原地操作时候，copy src到dst
    if(dst.data!=src.data)
    {
        src.copyTo(dst);
    }
    
    int i, j, n;
    int width, height;
    width = src.cols -1;
    //之所以减1，是方便处理8邻域，防止越界
    height = src.rows -1;
    int step = src.step;
    int  p2,p3,p4,p5,p6,p7,p8,p9;
    uchar* img;
    bool ifEnd;
    int A1;
    cv::Mat tmpimg;
    //n表示迭代次数
    for(n = 0; n<intera; n++)
    {
        dst.copyTo(tmpimg);
        ifEnd = false;
        img = tmpimg.data;
        for(i = 1; i < height; i++)
        {
            img += step;
            for(j =1; j<width; j++)
            {
                uchar* p = img + j;
                A1 = 0;
                if( p[0] > 0)
                {
                    if(p[-step]==0&&p[-step+1]>0) //p2,p3 01模式
                    {
                        A1++;
                    }
                    if(p[-step+1]==0&&p[1]>0) //p3,p4 01模式
                    {
                        A1++;
                    }
                    if(p[1]==0&&p[step+1]>0) //p4,p5 01模式
                    {
                        A1++;
                    }
                    if(p[step+1]==0&&p[step]>0) //p5,p6 01模式
                    {
                        A1++;
                    }
                    if(p[step]==0&&p[step-1]>0) //p6,p7 01模式
                    {
                        A1++;
                    }
                    if(p[step-1]==0&&p[-1]>0) //p7,p8 01模式
                    {
                        A1++;
                    }
                    if(p[-1]==0&&p[-step-1]>0) //p8,p9 01模式
                    {
                        A1++;
                    }
                    if(p[-step-1]==0&&p[-step]>0) //p9,p2 01模式
                    {
                        A1++;
                    }
                    p2 = p[-step]>0?1:0;
                    p3 = p[-step+1]>0?1:0;
                    p4 = p[1]>0?1:0;
                    p5 = p[step+1]>0?1:0;
                    p6 = p[step]>0?1:0;
                    p7 = p[step-1]>0?1:0;
                    p8 = p[-1]>0?1:0;
                    p9 = p[-step-1]>0?1:0;
                    if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7  &&  A1==1)
                    {
                        if((p2==0||p4==0||p6==0)&&(p4==0||p6==0||p8==0)) //p2*p4*p6=0 && p4*p6*p8==0
                        {
                            dst.at<uchar>(i,j) = 0; //满足删除条件，设置当前像素为0
                            ifEnd = true;
                        }
                    }
                }
            }
        }
        
        dst.copyTo(tmpimg);
        img = tmpimg.data;
        for(i = 1; i < height; i++)
        {
            img += step;
            for(j =1; j<width; j++)
            {
                A1 = 0;
                uchar* p = img + j;
                if( p[0] > 0)
                {
                    if(p[-step]==0&&p[-step+1]>0) //p2,p3 01模式
                    {
                        A1++;
                    }
                    if(p[-step+1]==0&&p[1]>0) //p3,p4 01模式
                    {
                        A1++;
                    }
                    if(p[1]==0&&p[step+1]>0) //p4,p5 01模式
                    {
                        A1++;
                    }
                    if(p[step+1]==0&&p[step]>0) //p5,p6 01模式
                    {
                        A1++;
                    }
                    if(p[step]==0&&p[step-1]>0) //p6,p7 01模式
                    {
                        A1++;
                    }
                    if(p[step-1]==0&&p[-1]>0) //p7,p8 01模式
                    {
                        A1++;
                    }
                    if(p[-1]==0&&p[-step-1]>0) //p8,p9 01模式
                    {
                        A1++;
                    }
                    if(p[-step-1]==0&&p[-step]>0) //p9,p2 01模式
                    {
                        A1++;
                    }
                    p2 = p[-step]>0?1:0;
                    p3 = p[-step+1]>0?1:0;
                    p4 = p[1]>0?1:0;
                    p5 = p[step+1]>0?1:0;
                    p6 = p[step]>0?1:0;
                    p7 = p[step-1]>0?1:0;
                    p8 = p[-1]>0?1:0;
                    p9 = p[-step-1]>0?1:0;
                    if((p2+p3+p4+p5+p6+p7+p8+p9)>1 && (p2+p3+p4+p5+p6+p7+p8+p9)<7  &&  A1==1)
                    {
                        if((p2==0||p4==0||p8==0)&&(p2==0||p6==0||p8==0)) //p2*p4*p8=0 && p2*p6*p8==0
                        {
                            dst.at<uchar>(i,j) = 0; //满足删除条件，设置当前像素为0
                            ifEnd = true;
                        }
                    }
                }
            }
        }
        
        //如果两个子迭代已经没有可以细化的像素了，则退出迭代
        if(!ifEnd) break;
    }
    
}
extern Mat* save_img(Mat,char *,string);

int main(int argc, char * argv[])
{
    Mat src = imread(argv[1]);
    if (src.empty())
        return -1;

    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    Mat dilate1;
    Mat ske;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));// MORPH_RECT,MORPH_ELLIPSE,MORPH_CROSS
    dilate(bw, dilate1, element);
    
    cvThin(dilate1,ske,15);
    
    string add="-ske.tif";
    save_img(ske,argv[1],add);
    
    return 0;
}


