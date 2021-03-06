//
//  main.cpp
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

extern Mat morph_ske(Mat);
extern Mat* save_img(Mat,char *,string);

int main(int argc, char *argv[])
{
    Mat img=imread(argv[1],0);
    //    Mat img=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Test/RegionMethod/三步对比/Results-filter5/北方角毛藻/北方角毛藻_环面观_南海_00000000_115拖网_040_00_resize_svm-5maps-filter.tif",0);
    Mat skel=morph_ske(img);
    //    imshow("skel",skel);
    
    string add="-ske.tif";
    save_img(skel,argv[1],add);
    
    //    imshow("Skeleton", skel);
    //    waitKey(0);
    return 0;
}
