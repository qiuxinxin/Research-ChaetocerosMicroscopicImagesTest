//
//  test.cpp
//  cluster_test
//
//  Created by qiuxinxin on 15/6/29.
//  Copyright (c) 2015年 qiu. All rights reserved.
//

//#include <iostream>
//#include <opencv/cv.h>
//#include <opencv/highgui.h>
//#include <stdio.h>
//#include <string.h>
//#include <string>
//#include <opencv/cxcore.hpp>
//
//using namespace cv;
//using namespace std;
//
//
//IplImage *image_x=cvLoadImage("./1.jpg",0);
//cv::Mat imagex=cv::cvarrToMat(image_x);

//    Scalar colorTab[] =
//    {
//        Scalar(0, 0, 255),
//        Scalar(0,255,0)
//    };
//    for( i = 0; i < imagex.cols*imagex.rows; i++ )
//    {
//        int clusterIdx = labels1.at<int>(i);
//        printf("id=%d\n",clusterIdx);
//        Point ipt = samples.at<Point2f>(i);
//        cvCircle(&image, ipt, 2, colorTab[clusterIdx], CV_FILLED, CV_AA );
//    }
//    Mat img1(image.size(), CV_8UC3);
//    img1 = Scalar::all(0);
//    cvNamedWindow("K-means++");
//    cvShowImage( "clusters", &image );
//    cvWaitKey(0);

//colorTab.push_back(Vec3b(0,0,0));
////    colorTab.push_back(Vec3b(255,0,0));
//
//int m=0;
//for( i = 0; i < image.rows; i++ )
//for( j = 0; j < image.cols; j++ )
//{
//    int index = labels1.at<int>(m,0);
//    //            printf("in=%d\n",index);
//    m++;
//    if( index == -1 )
//        img1.at<Vec3b>(i,j) = Vec3b(255,0,0);
//        else if( index <= 0 || index > cluster_num )
//            img1.at<Vec3b>(i,j) = Vec3b(255,255,255);
//            else
//                img1.at<Vec3b>(i,j) = colorTab[index-1];
//                //             cout<<colorTab[index-1]<<"\n";

//Mat imagex=cv::cvarrToMat(image_x);//iplimage转mat
//Mat imagey=cv::cvarrToMat(image_y);
//Mat imagez=cv::cvarrToMat(image_z);
//Mat imagexz=cv::cvarrToMat(image_xz);
//Mat imageyz=cv::cvarrToMat(image_yz);
//Mat image=cv::cvarrToMat(src);
//Mat samples(imagex.cols*imagex.rows, 1, CV_32FC4);
//Mat labels1(imagex.cols*imagex.rows, 1, CV_32SC1);
//
//uchar *p1, *p2, *p3, *p4;;
//int i, j, k=0;
//for(i=0; i < imagex.rows; i++)
//{
//    p1 = imagex.ptr<uchar>(i);
//    p2 = imagey.ptr<uchar>(i);
//    p3 = imagez.ptr<uchar>(i);
//    p4 = imagexz.ptr<uchar>(i);
//    for(j=0; j< imagex.cols; j++)
//    {
//        samples.at<Vec3f>(k,0)[0] = float(p1[j]);
//        samples.at<Vec3f>(k,0)[1] = float(p2[j]);
//        samples.at<Vec3f>(k,0)[2] = float(p3[j]);
//        samples.at<Vec3f>(k,0)[4] = float(p4[j]);
//        //            cout<<samples.at<Vec3f>(k,0)[0]<<"\n";
//        k++;
//    }
//}
//
//int cluster_num=2;
//Mat centers1(cluster_num, 1, samples.type());//用来存储聚类后的中心点
//kmeans(samples, cluster_num, labels1, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 10, 1.0), 1,KMEANS_PP_CENTERS, centers1);//TermCriteria控制迭代算法的终止条件
//
//Mat img1(image.size(), CV_8UC3);
//vector<Vec3b> colorTab;

//    for( i = 0; i < cluster_num; i++ )
//    {
//        int b = theRNG().uniform(0, 255);//由RNG类型变量产生的随机数，范围在【0-255】之间
//        int g = theRNG().uniform(0, 255);
//        int r = theRNG().uniform(0, 255);
//        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));//push_back作用是在这个一维数组尾部插入一个元素 其值为push_back的参数值
//        cout<<colorTab[0]<<"\n";
//        cout<<colorTab[1]<<"\n";
//    }

//    colorTab.push_back(Vec3b(0,0,0));
//    colorTab.push_back(Vec3b(255,0,0));

//int m=0;
//for( i = 0; i < image.rows; i++ )
//for( j = 0; j < image.cols; j++ )
//{
//    int index = labels1.at<int>(m,0);
//    // printf("in=%d\n",index);
//    m++;
//    if( index == -1 )
//        img1.at<Vec3b>(i,j) = Vec3b(255,0,0);
//        else if (index < 0 || index > cluster_num)
//            img1.at<Vec3b>(i,j) = Vec3b(0,255,0);
//            else if (index==0)
//                img1.at<Vec3b>(i,j) = Vec3b(255,255,255);
//                else
//                    img1.at<Vec3b>(i,j) = Vec3b(0,0,0);
//                    }
//
//imwrite("./kmeans++2.tif",img1);
//
//namedWindow("K-means++");
//imshow("K-means++", img1);
//waitKey();
//
//cvReleaseImage(&image_x);
//cvReleaseImage(&image_y);
//cvReleaseImage(&image_z);
//
//return 0;
//}


//                }

//#include <iostream>
//#include <opencv2/highgui.hpp>
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include <string>
//
//
//using namespace cv;
//using namespace std;
//typedef IplImage* IPL;
//
//extern double compute_direction_angles(double,double,double,double,double *,double *,double *,double *);
//extern IPL save_img(IPL,char *,string);
//
//
//int main(int argc, char* argv[])
//{
//    IPL src1=cvLoadImage(argv[1],1);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像,1为RGB图像
//    //    IPL src = cvCreateImage( cvSize( src1 -> width+2, src1 -> height+2 ), IPL_DEPTH_8U, src1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称
//    //    IPL image_x=cvCloneImage(src);
//    //    IPL image_y=cvCloneImage(src);
//    //    IPL image_z=cvCloneImage(src);
//    //    IPL image_xy=cvCloneImage(src);
//    //    IPL image_xz=cvCloneImage(src);
//    //    IPL image_yz=cvCloneImage(src);
//    //
//    //    int height=src->height, width=src->width;
//    //    //    printf("h=%d w=%d\n",height,width);
//    //    double **c1;
//    //    double **c2;
//    //    double **c3;
//    //    double **c4;
//    //    double **xmap;
//    //    double **ymap;
//    //    double **zmap;
//    //    double **xymap;
//    //    double **xzmap;
//    //    double **yzmap;
//    //
//    //    c1 = new double*[width];//x方向角的角度 new表示动态内存分配 c1作为一个指向指针的指针，它指向一个包含width个元素的指针数组
//    //    c2 = new double*[width];
//    //    c3 = new double*[width];
//    //    c4 = new double*[width];
//    //    xmap=new double*[width];
//    //    ymap=new double*[width];
//    //    zmap=new double*[width];
//    //    xymap=new double*[width];
//    //    xzmap=new double*[width];
//    //    yzmap=new double*[width];
//    //
//    //
//    //    cvCopyMakeBorder(src1, src, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
//    //
//    //    for(int i=0; i<width; i++)
//    //    {
//    //        c1[i] = new double[height];//每个指针元素指向一个有height个元素的数组，这样就构建了有width行height列的数组
//    //        c2[i] = new double[height];
//    //        c3[i] = new double[height];
//    //        c4[i] = new double[height];
//    //        xmap[i]=new double[height];
//    //        ymap[i]=new double[height];
//    //        zmap[i]=new double[height];
//    //        xymap[i]=new double[height];
//    //        xzmap[i]=new double[height];
//    //        yzmap[i]=new double[height];
//    //
//    //    }
//    //
//    //    int x,y;
//    //    CvScalar Ix0y0,Ix0y1,Ix1y1,Ix1y0;//cvscalar的结构为一个double型的4成员的数组，它提供了四个成员，最多就可对应四个通道
//    //    double w1,w2,w3,* w1_z,* w2_z,* w3_z,dis,* dis_z;
//    //
//    //    for(y=0; y<height-1; y++)
//    //    {
//    //        for(x=0; x<width-1; x++)
//    //        {
//    //            Ix0y0=cvGet2D(src,y,x);//提取图像(y,x)的像素值，与通常坐标系相反
//    //            //            printf("y=%d x=%d ixoyo=%f\n",y,x,Ix0y0.val[0]);
//    //            Ix0y1=cvGet2D(src,(y+1),x);
//    //            Ix1y1=cvGet2D(src,(y+1),(x+1));
//    //            //            printf("y=%d x=%d ixoyo=%f\n",y+1,x+1,Ix1y1.val[0]);
//    //            Ix1y0=cvGet2D(src,y,(x+1));
//    //            w1_z=&w1;
//    //            w2_z=&w2;
//    //            w3_z=&w3;
//    //            dis_z=&dis;
//    //            compute_direction_angles(Ix0y0.val[0],Ix0y1.val[0],Ix1y0.val[0],Ix1y1.val[0],w1_z,w2_z,w3_z,dis_z);
//    //            //此函数作用是分别求出xyz方向的三个夹角大小
//    //            c1[x][y]=w1;//w1是法向量与与x轴的夹角；
//    //            c2[x][y]=w2;
//    //            c3[x][y]=w3;
//    //        }
//    //    }
//    //
//    //    double xmax,xmin,ymax,ymin,zmax,zmin;
//    //    double c1yx,c2yx,c3yx;
//    //    xmax=c1[0][0];
//    //    xmin=c1[0][0];
//    //    ymax=c2[0][0];
//    //    ymin=c2[0][0];
//    //    zmax=c3[0][0];
//    //    zmin=c3[0][0];
//    //
//    //    for(y=0; y<height-1; y++)
//    //    {
//    //        for(x=0; x<width-1; x++)
//    //        {
//    //            c1yx=c1[x][y];
//    //            if (c1yx>xmax)
//    //                xmax=c1yx;//max(theta_x)
//    //            else {};
//    //            if (c1yx<xmin)
//    //                xmin=c1yx;//min(theta_x)
//    //            else {};
//    //
//    //            c2yx=c2[x][y];
//    //            if (c2yx>ymax)
//    //                ymax=c2yx;
//    //            else {};
//    //            if (c2yx<ymin)
//    //                ymin=c2yx;
//    //            else {};
//    //
//    //            c3yx=c3[x][y];
//    //            if (c3yx>zmax)
//    //                zmax=c3yx;
//    //            else {};
//    //            if (c3yx<zmin)
//    //                zmin=c3yx;
//    //            else {};
//    //        }
//    //    }
//    //    for(y=0; y<height-1; y++)
//    //    {
//    //        for(x=0; x<width-1; x++)
//    //        {
//    //            xmap[x][y]=255*(c1[x][y]-xmin)/(xmax-xmin);
//    //            ymap[x][y]=255*(c2[x][y]-ymin)/(ymax-ymin);
//    //            zmap[x][y]=255*(c3[x][y]-zmin)/(zmax-zmin);
//    //            //            printf("x=%f\n",xmap[x][y]);
//    //            cvSet2D(image_x,y,x,cvRealScalar(xmap[x][y]));//将map的值赋给图像中每个像素点
//    //            cvSet2D(image_y,y,x,cvRealScalar(ymap[x][y]));//cvrealscalar把数组的第一个值置为想要的数（即为 double val0 ），其他三个数均为零。
//    //            cvSet2D(image_z,y,x,cvRealScalar(zmap[x][y]));
//    //
//    //        }
//    //    }
//    //
//    //
//    //    for(y=0; y<height-1; y++)
//    //    {
//    //        for(x=0; x<width-1; x++)
//    //        {
//    //            xymap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+ymap[x][y]*ymap[x][y]);
//    //            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
//    //            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
//    //            cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
//    //            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
//    //            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
//    //        }
//    //    }
//    //    //
//    //    double xymap_max=xymap[0][0];
//    //    double xymap_min=xymap[0][0];
//    //    double xzmap_max=xzmap[0][0];
//    //    double xzmap_min=xzmap[0][0];
//    //    double yzmap_max=yzmap[0][0];
//    //    double yzmap_min=yzmap[0][0];
//    //    for(y=0; y<height-1; y++)
//    //    {
//    //        for(x=0; x<width-1; x++)
//    //        {
//    //            if (xymap[x][y]>xymap_max)
//    //            {
//    //                xymap_max=xymap[x][y];
//    //            }
//    //            else {}
//    //            if (xymap[x][y]<xymap_min)
//    //            {
//    //                xymap_min=xymap[x][y];
//    //            }
//    //            else {}
//    //            if (xzmap[x][y]>xzmap_max)
//    //            {
//    //                xzmap_max=xzmap[x][y];
//    //            }
//    //            else {}
//    //            if (xzmap[x][y]<xzmap_min)
//    //            {
//    //                xzmap_min=xzmap[x][y];
//    //            }
//    //            else {}
//    //            if (yzmap[x][y]>yzmap_max)
//    //            {
//    //                yzmap_max=yzmap[x][y];
//    //            }
//    //            else {}
//    //            if (yzmap[x][y]<yzmap_min)
//    //            {
//    //                yzmap_min=yzmap[x][y];
//    //            }
//    //            else {}
//    //        }
//    //    }
//    //
//    //    for(y=0; y<height-1; y++)
//    //    {
//    //        for(x=0; x<width-1; x++)
//    //        {
//    //            xymap[x][y]=255*( (xymap[x][y]-xymap_min)/(xymap_max-xymap_min));
//    //            //               printf("xy=%f\n",xymap[x][y]);
//    //            xzmap[x][y]=255*( (xzmap[x][y]-xzmap_min)/(xzmap_max-xzmap_min));//对xzmap,yzmap进行255归一化
//    //            yzmap[x][y]=255*( (yzmap[x][y]-yzmap_min)/(yzmap_max-yzmap_min));
//    //            //                printf("y=%d x=%d xxx=%f yyy=%f zzz=%f xxy=%f xxz=%f xyz=%f\n",y,x,xmap[x][y],ymap[x][y],zmap[x][y],xymap[x][y],xzmap[x][y],yzmap[x][y]);
//    //            cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
//    //            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
//    //            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
//    //        }
//    //    }
//    //
//    //    for(int i=0; i<width; i++)
//    //    {
//    //        delete []c1[i];//先撤销指针元素所指向的数组
//    //        delete []c2[i];
//    //        delete []c3[i];
//    //        delete []c4[i];
//    //        delete []xmap[i];
//    //        delete []ymap[i];
//    //        delete []zmap[i];
//    //        delete []xymap[i];
//    //        delete []xzmap[i];
//    //        delete []yzmap[i];
//    //    }
//    //    delete []c1;//在销毁的过程，先销毁指针数组每个元素指向的数组，然后再销毁这个指针数组
//    //    delete []c2;
//    //    delete []c3;
//    //    delete []c4;
//    //    delete []xmap;
//    //    delete []ymap;
//    //    delete []zmap;
//    //    delete []xymap;
//    //    delete []xzmap;
//    //    delete []yzmap;
//
//    //    Mat src3=cvarrToMat(src);
//    //    Mat temp;
//    //    src3.copyTo(temp);
//    //    Mat image=temp.reshape(1,height*width);
//    //    cout<<image<<endl;
//    //    CvScalar ixy;
//    int i,j;
//
//    //    Mat image(1,height*width,CV_32FC3);
//    //
//    //    for (i=0;i<height;i++)
//    //        for (j=0;j<width;j++)
//    //        {
//    //            ixy=cvGet2D(src, i, j);
//    //            cvSet2D(&image, i+j+(width-1)*i, 1, ixy);
//    //        }
//
//
//    //    Mat imagex=cvarrToMat(image_x);//iplimage转mat
//    ////        cout<<imagex<<endl;
//    //    Mat imagey=cvarrToMat(image_y);
//    //    Mat imagez=cvarrToMat(image_z);
//    //    Mat imagexz=cvarrToMat(image_xz);
//    //    Mat imageyz=cvarrToMat(image_yz);
//    Mat image=cvarrToMat(src1);
//    //    cout<<image<<endl;
//    //    Mat image=imread(argv[1]);
//    //    Mat tempx,tempy,tempz,tempxz,tempyz;//避免直接使用出现矩阵不连续的错误
//    //    imagex.copyTo(tempx);
//    //    imagey.copyTo(tempy);
//    //    imagez.copyTo(tempz);
//    //    imagexz.copyTo(tempxz);
//    //    imageyz.copyTo(tempyz);
//    //
//    //    Mat imgx=tempx.reshape(1,height*width);//将图像像素排成一列
//    //    //    cout<<imgx.at<uchar>()<<endl;
//    //    Mat imgy=tempy.reshape(1,height*width);
//    //    Mat imgz=tempz.reshape(1,height*width);
//    //    Mat imgxz=tempxz.reshape(1,height*width);
//    //    Mat imgyz=tempyz.reshape(1,height*width);
//    int height=image.rows;
//    int width=image.cols;
//    //    cout<<height<<"\n"<<width<<endl;
//    Mat sample(height*width,1,CV_32FC3);//生成height*width行5列矩阵，注意采样点格式为32bit浮点数
//    Mat labels1(height*width, 1, CV_32SC1);
//    Mat img1(image.size(), CV_8UC3);
//
//
//    //    for(i=0; i < height*width; i++)
//    //    {
//    //        sample.at<Vec3f>(i,0)=imgx.at<Vec3b>(i,1);//一行为一个样本（即某个像素点），每列为一个特征
//    //        //        cout<<sample.at<Vec3f>(i,0)<<"\n";
//    //        sample.at<Vec3f>(i,1)=imgy.at<Vec3b>(i,1);
//    //        sample.at<Vec3f>(i,2)=imgz.at<Vec3b>(i,1);
//    //        sample.at<Vec3f>(i,3)=imgxz.at<Vec3b>(i,1);
//    //        sample.at<Vec3f>(i,4)=imgyz.at<Vec3b>(i,1);
//    //        sample.at<Vec3f>(i,5)=image.at<Vec3b>(i,1)[0];
//    //        cout<<image.at<Vec3b>(i,1)(1)<<endl;
//    //        sample.at<Vec3f>(i,6)=image.at<Vec3b>(i,1)[1];
//    //        sample.at<Vec3f>(i,7)=image.at<Vec3b>(i,1)[2];
//    //    }
//
//    //    uchar* p;
//    //    for(i=0; i < height; i++)
//    //    {
//    //        p = image.ptr<uchar>(i);
//    //        //        cout<<p<<endl;
//    //        for(j=0; j< width; j++)
//    //        {
//    //            sample.at<Vec3f>(i*width+j,5)= float(p[j*3]);
//    //            sample.at<Vec3f>(i*width+j,6)= float(p[j*3+1]);
//    //            sample.at<Vec3f>(i*width+j,7)= float(p[j*3+2]);
//    ////            cout<<float(p[j*3])<<endl;
//    //        }
//    //    }
//
//
//
//    uchar* p;
//    int k=0;
//    for(i=0; i < height; i++)
//    {
//        p = image.ptr<uchar>(i);
//        //      cout<<p<<endl;
//        for(j=0; j< width; j++)
//        {
//            sample.at<Vec3f>(k,0)[0] = float(p[j*3]);
//            sample.at<Vec3f>(k,0)[1] = float(p[j*3+1]);
//            sample.at<Vec3f>(k,0)[2] = float(p[j*3+2]);
//            //            cout<<float(p[j*3+1])<<endl;
//            k++;
//        }
//    }
//
//    int cluster_num=2;
//    Mat centers1(cluster_num, 1, sample.type());//用来存储聚类后的中心点
//    kmeans(sample, cluster_num, labels1, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 10, 1.0), 1,KMEANS_PP_CENTERS, centers1);//TermCriteria控制迭代算法的终止条件
//
//    int m=0;
//    for( i = 0; i < image.rows; i++ )
//        for( j = 0; j < image.cols; j++ )
//        {
//            int index = labels1.at<int>(m,0);
//            //             printf("in=%d\n",index);
//            m++;
//            if( index == -1 )
//                img1.at<Vec3b>(i,j) = Vec3b(255,0,0);
//            else if (index < 0 || index > cluster_num)
//                img1.at<Vec3b>(i,j) = Vec3b(0,255,0);
//            else if (index==0)
//                img1.at<Vec3b>(i,j) = Vec3b(255,255,255);
//            else
//                img1.at<Vec3b>(i,j) = Vec3b(0,0,0);
//        }
//    //
//    string savingfile(argv[1]);
//    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
//    //    printf("str1=%s\n",shortstring1.c_str());
//    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
//    string add="_kmeans++3feature.tif";
//    shortstring+=add;//加入新的标示及扩展名
//    imwrite(shortstring,img1);
//
//    //        namedWindow("K-means++");
//    //        imshow("K-means++", img1);
//    //        waitKey();
//    //
//    //    cvReleaseImage(&image_x);
//    //    cvReleaseImage(&image_y);
//    //    cvReleaseImage(&image_z);
//    //    cvReleaseImage(&image_xz);
//    //    cvReleaseImage(&image_yz);
//
//    return 0;
//}



