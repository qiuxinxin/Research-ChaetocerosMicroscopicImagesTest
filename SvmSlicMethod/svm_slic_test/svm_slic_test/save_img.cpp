//
//  save_img.cpp
//  skeleton
//
//  Created by qiuxinxin on 15/8/7.
//  Copyright (c) 2015年 qiu. All rights reserved.
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Mat* save_img(Mat img,char *string_img,string add)
{
    string savingfile(string_img);
    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
    shortstring+=add;//加入新的标示及扩展名
    imwrite(shortstring,img);
    return NULL;
}