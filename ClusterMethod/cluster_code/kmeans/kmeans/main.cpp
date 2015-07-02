#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( int /*argc*/, char** /*argv*/ )
{
    Mat img = cv::imread("/Users/qiuxinxin/temp/角毛藻显微图像/cluster/code/kmeans/kmeans/3.jpg");
    namedWindow("image");
    imshow("image", img);
    //生成一维采样点,包括所有图像像素点,注意采样点格式为32bit浮点数。
    Mat samples(img.cols*img.rows, 1, CV_32FC3);
    //标记矩阵，32位整形
    Mat labels1(img.cols*img.rows, 1, CV_32SC1);
    Mat labels2(img.cols*img.rows, 1, CV_32SC1);
    
    uchar* p;
    int i, j, k=0;
    for(i=0; i < img.rows; i++)
    {
        p = img.ptr<uchar>(i);
//        cout<<p<<endl;
        for(j=0; j< img.cols; j++)
        {
            samples.at<Vec3f>(k,0)[0] = float(p[j*3]);
            samples.at<Vec3f>(k,0)[1] = float(p[j*3+1]);
            samples.at<Vec3f>(k,0)[2] = float(p[j*3+2]);
            cout<<float(p[j*3])<<endl;
            k++;
        }
    }
    
    int clusterCount = 2;
    Mat centers1(clusterCount, 1, samples.type());
    Mat centers2(clusterCount, 1, samples.type());
    kmeans(samples, clusterCount, labels1, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 10, 1.0), 1,KMEANS_PP_CENTERS, centers1);//TermCriteria控制迭代算法的终止条件
    kmeans(samples, clusterCount, labels2, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 10, 1.0), 1,KMEANS_RANDOM_CENTERS, centers2);//TermCriteria控制迭代算法的终止条件,随机选取聚类中心
    
    //我们已知有6个聚类，用不同的颜色表示。
    Mat img1(img.size(), CV_8UC3);
    Mat img2(img.size(), CV_8UC3);
    vector<Vec3b> colorTab;
    int m=0;
    for( i = 0; i < clusterCount; i++ )
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        
        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    for( i = 0; i < img.rows; i++ )
        for( j = 0; j < img.cols; j++ )
        {
            int index = labels1.at<int>(m,0);
            m++;
            if( index == -1 )
                img1.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else if( index <= 0 || index > clusterCount )
                img1.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                img1.at<Vec3b>(i,j) = colorTab[index - 1];
        }
    int n=0;
    for( i = 0; i < img.rows; i++ )
        for( j = 0; j < img.cols; j++ )
        {
            int index = labels2.at<int>(n,0);
            n++;
            if( index == -1 )
                img2.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else if( index <= 0 || index > clusterCount )
                img2.at<Vec3b>(i,j) = Vec3b(0,0,0);
            else
                img2.at<Vec3b>(i,j) = colorTab[index - 1];
        }

    namedWindow("K-means++");
    imshow("K-means++", img1);
    waitKey();
//    namedWindow("K-means");
//    imshow("K-means", img2);
//    waitKey();
    return 0;
}
