//
//  thinningGuoHallIteration.cpp
//  skeleton
//
//  Created by qiuxinxin on 15/8/6.
//  Copyright (c) 2015年 qiu. All rights reserved.
//

/**
 * Code for thinning a binary image using Guo-Hall algorithm.
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void thinningGuoHallIteration(Mat& im, int iter)
{
    Mat marker = Mat::zeros(im.size(), CV_8UC1);
    
    for (int i = 1; i < im.rows; i++)
    {
        for (int j = 1; j < im.cols; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);
            
            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
            (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);
            
            if (C == 1 && (N >= 2 && N <= 3) & (m == 0))
                marker.at<uchar>(i,j) = 1;
        }
    }
    
    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinningGuoHall(Mat& im)
{
    im /= 255;
    
    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;
    
    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (countNonZero(diff) > 0);
    
    im *= 255;
}

/**
 * This is an example on how to call the thinning function above.
 */
extern Mat* save_img(Mat,char *,string);

int main(int argc, char * argv[])
{
    Mat src = imread(argv[1]);
    if (src.empty())
        return -1;
    
//    cout<<src<<endl;
    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
//    cout<<bw<<endl;
    Mat dilate1;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));// MORPH_RECT,MORPH_ELLIPSE,MORPH_CROSS
    dilate(bw, dilate1, element);
    
//    imshow("src", src);
//    imshow("dil", dilate1);
    
    thinningGuoHall(dilate1);
    
//    string savingfile("/Users/qiuxinxin/temp/角毛藻显微图像/Test/RegionMethod/三步对比/Results-filter5/丹麦角毛藻/丹麦角毛藻_壳面观_青岛沿海_20041210_00_040_00_resize_svm-5maps-filter.tif");
//    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
//    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
//    string add="-ske.tif";
//    shortstring+=add;//加入新的标示及扩展名
//    imwrite(shortstring,dilate1);

    string add="-ske.tif";
    save_img(dilate1,argv[1],add);
    
//    imshow("dst", dilate1);
//    waitKey();
    return 0;
}
