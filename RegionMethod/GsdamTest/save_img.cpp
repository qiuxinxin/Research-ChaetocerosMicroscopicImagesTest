#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv/cxcore.h"
#include <iostream>
#include<string>
using namespace std;

IplImage* save_img(IplImage* pImg, char *sring_src, string add)
{
    string savingfile(sring_src);
    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
//    printf("str1=%s\n",shortstring1.c_str());
    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
    shortstring+=add;//加入新的标示及扩展名
    cvSaveImage(shortstring.c_str(),pImg);//路径
    return NULL;
}
