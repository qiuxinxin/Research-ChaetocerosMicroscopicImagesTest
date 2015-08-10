//
//  morph_ske.cpp
//  skeleton
//
//  Created by qiuxinxin on 15/8/6.
//  Copyright (c) 2015年 qiu. All rights reserved.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

Mat morph_ske(Mat img)
{
    threshold(img, img, 127, 255, THRESH_BINARY);
    Mat skel(img.size(), CV_8UC1, Scalar(0));
    Mat temp;
    Mat eroded;
    
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    
    bool done;
    do
    {
        erode(img, eroded, element);
        dilate(eroded, temp, element); // temp = open(img)
        subtract(img, temp, temp);
        bitwise_or(skel, temp, skel);
        eroded.copyTo(img);
        
        done = (countNonZero(img) == 0);
    } while (!done);

    return skel;
}

