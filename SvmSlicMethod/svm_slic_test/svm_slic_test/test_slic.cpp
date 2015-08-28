#include "slic.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

//*****存储图像********//
void save_img (Mat img, char *string_img, string add)
{
    string savingfile(string_img);
    cout<<string_img<<endl;
    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
    shortstring+=add;//加入新的标示及扩展名
    imwrite(shortstring,img);
}
//********计算x、y、z方向的方向角************//
void compute_direction_angles(double p1,double p2,double p3,double p4,double * w_1,double * w_2,double * w_3,double * dis)
{
    const  double  PI=3.1415926;
    
    double f1_1,f1_2,f1_3,f2_1,f2_2,f2_3,g1_1,g1_2,g1_3,g2_1,g2_2,g2_3,h1_1,h1_2,h1_3,h2_1,h2_2,h2_3,m1_1,m1_2,m1_3,m2_1,m2_2,m2_3;
    double sf_1,sf_2,sf_3,sg_1,sg_2,sg_3,sh_1,sh_2,sh_3,sm_1,sm_2,sm_3;
    double v1_1,v1_2,v1_3,v2_1,v2_2,v2_3,v_1,v_2,v_3;
    f1_1=1; f1_2=0; f1_3=p3-p1;
    f2_1=0; f2_2=1; f2_3=p2-p1;
    
    sf_1=f1_2*f2_3-f1_3*f2_2; //x方向矢量,为论文中f_{ACB}=AC X AB叉乘行列式的i方向
    sf_2=f1_3*f2_1-f1_1*f2_3; //y方向矢量
    sf_3=f1_1*f2_2-f1_2*f2_1; //z方向矢量
    //以上为点1,2,3组成的面的法线矢量，(左上角1，右上角2，左下角3，右下角4)
    g1_1=0;  g1_2=-1;  g1_3=p1-p2;
    g2_1=1;  g2_2=0;   g2_3=p4-p2;
    
    sg_1=g1_2*g2_3-g1_3*g2_2;//BAD平面
    sg_2=g1_3*g2_1-g1_1*g2_3;
    sg_3=g1_1*g2_2-g1_2*g2_1;
    //点1,2,4,组成的面的法线矢量
    h1_1=-1; h1_2=0;  h1_3=p2-p4;
    h2_1=0;  h2_2=-1; h2_3=p3-p4;
    
    sh_1=h1_2*h2_3-h1_3*h2_2;
    sh_2=h1_3*h2_1-h1_1*h2_3;
    sh_3=h1_1*h2_2-h1_2*h2_1;
    //点4,2,3,组成的面的法线矢量
    
    m1_1=0;  m1_2=1; m1_3=p4-p3;
    m2_1=-1; m2_2=0; m2_3=p1-p3;
    
    sm_1=m1_2*m2_3-m1_3*m2_2;
    sm_2=m1_3*m2_1-m1_1*m2_3;
    sm_3=m1_1*m2_2-m1_2*m2_1;
    //点1,4,3,组成的面的法线矢量
    
    v1_1=sf_1+sh_1;v1_2=sf_2+sh_2;v1_3=sf_3+sh_3;
    v2_1=sg_1+sm_1;v2_2=sg_2+sm_2;v2_3=sg_3+sm_3;
    
    v_1=(v1_1+v2_1)/4;v_2=(v1_2+v2_2)/4;v_3=(v1_3+v2_3)/4;//f_{A}=(f_{ACB}+f_{BAD}+f_{DBC}+f_{CDA})/4
    double dis2=sqrt(v_1*v_1+v_2*v_2+v_3*v_3);
    *dis=dis2;
    
    double w_12=360*acos(v_1/dis2)/(2*PI);
    double w_22=360*acos(v_2/dis2)/(2*PI);
    double w_32=360*acos(v_3/dis2)/(2*PI);
    
    *w_1=w_12;
    *w_2=w_22;
    *w_3=w_32;
}
//************计算5个map**********//
void gsdam(IplImage* image1, Mat image2, double **xmap,double **ymap,double **zmap, double **xzmap, double **yzmap)
{
    IplImage* image_border = cvCreateImage( cvSize( image1 -> width+2, image1 -> height+2 ), IPL_DEPTH_8U, image1->nChannels);
    
    int height=image_border->height, width=image_border->width;
    double **c1;
    double **c2;
    double **c3;
    double **c4;
    
    c1 = new double*[width];
    c2 = new double*[width];
    c3 = new double*[width];
    c4 = new double*[width];
    
    cvCopyMakeBorder(image1, image_border, cvPoint(1,1),IPL_BORDER_CONSTANT);
    
    for(int i=0; i<width; i++)
    {
        c1[i] = new double[height];
        c2[i] = new double[height];
        c3[i] = new double[height];
        c4[i] = new double[height];
    }
    
    int x,y;
    CvScalar Ix0y0,Ix0y1,Ix1y1,Ix1y0;
    double w1,w2,w3,* w1_z,* w2_z,* w3_z,dis,* dis_z;
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            Ix0y0=cvGet2D(image_border,y,x);
            Ix0y1=cvGet2D(image_border,(y+1),x);
            Ix1y1=cvGet2D(image_border,(y+1),(x+1));
            Ix1y0=cvGet2D(image_border,y,(x+1));
            w1_z=&w1;
            w2_z=&w2;
            w3_z=&w3;
            dis_z=&dis;
            compute_direction_angles(Ix0y0.val[0],Ix0y1.val[0],Ix1y0.val[0],Ix1y1.val[0],w1_z,w2_z,w3_z,dis_z);
            //此函数作用是分别求出xyz方向的三个夹角大小
            c1[x][y]=w1;//w1是法向量与与x轴的夹角；
            c2[x][y]=w2;
            c3[x][y]=w3;
        }
    }
    
    double xmax,xmin,ymax,ymin,zmax,zmin;
    double c1yx,c2yx,c3yx;
    xmax=c1[0][0];
    xmin=c1[0][0];
    ymax=c2[0][0];
    ymin=c2[0][0];
    zmax=c3[0][0];
    zmin=c3[0][0];
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            c1yx=c1[x][y];
            if (c1yx>xmax)
            xmax=c1yx;//max(theta_x)
            else {};
            if (c1yx<xmin)
            xmin=c1yx;//min(theta_x)
            else {};
            
            c2yx=c2[x][y];
            if (c2yx>ymax)
            ymax=c2yx;
            else {};
            if (c2yx<ymin)
            ymin=c2yx;
            else {};
            
            c3yx=c3[x][y];
            if (c3yx>zmax)
            zmax=c3yx;
            else {};
            if (c3yx<zmin)
            zmin=c3yx;
            else {};
        }
    }
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xmap[x][y]=255*(c1[x][y]-xmin)/(xmax-xmin);
            ymap[x][y]=255*(c2[x][y]-ymin)/(ymax-ymin);
            zmap[x][y]=255*(c3[x][y]-zmin)/(zmax-zmin);
        }
    }
    
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
        }
    }
    
    double xzmap_max=xzmap[0][0];
    double xzmap_min=xzmap[0][0];
    double yzmap_max=yzmap[0][0];
    double yzmap_min=yzmap[0][0];
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            if (xzmap[x][y]>xzmap_max)
            {
                xzmap_max=xzmap[x][y];
            }
            else {}
            if (xzmap[x][y]<xzmap_min)
            {
                xzmap_min=xzmap[x][y];
            }
            else {}
            if (yzmap[x][y]>yzmap_max)
            {
                yzmap_max=yzmap[x][y];
            }
            else {}
            if (yzmap[x][y]<yzmap_min)
            {
                yzmap_min=yzmap[x][y];
            }
            else {}
        }
    }
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xzmap[x][y]=255*((xzmap[x][y]-xzmap_min)/(xzmap_max-xzmap_min));
            yzmap[x][y]=255*((yzmap[x][y]-yzmap_min)/(yzmap_max-yzmap_min));
        }
    }
}
//*********svm训练与预测**********//
void svm(Mat img_seg,double **xmap,double **ymap,double **zmap,double **xzmap,double **yzmap,Mat img_new1,double maxid)
{
    /****记下所有区域内分割像素数占区域所有像素数大于等于一半的区域编号****/
    int i,j,m,regioncount=0;
    Mat labelMat(0,1,CV_32FC1);
    Mat trainingData1(0,1,CV_32FC1);//正样本
    Mat trainingDataMat1;
    
    Mat seg_count=Mat::zeros(Size(1,maxid+1),CV_32S);//size，前面为列，后面为行
    Mat img_count=Mat::zeros(Size(1,maxid+1),CV_32S);
    vector<Mat> pixelMat;//记录每个区域内的像素坐标
    Mat temp(0,1,CV_32S);
    for ( m = 0; m < maxid+1; m++)
    {
        pixelMat.push_back(temp);
    }
    //    cout<<pixelMat.size()<<endl;
    for( i = 0; i < img_new1.rows; i++)
    {
        for ( j = 0; j< img_new1.cols; j++)
        {
            for ( m = 0; m < maxid+1; m++)
            {
                if (img_new1.at<float>(i,j)==m)
                {
                    pixelMat[m].push_back(i);//横纵坐标交替写入
                    pixelMat[m].push_back(j);
                    img_count.at<int>(m,0)++;
                    //                    cout<<img_count.at<int>(m,0)<<endl;
                    if (img_seg.at<float>(i+1,j+1)==1)
                    {
                        seg_count.at<int>(m,0)++;
                    }
                    continue;
                }
            }
        }
    }
    //    cout<<pixelMat[0]<<endl;
    for (m=0; m<maxid+1; m++)
    {
        //        cout<<seg_count.at<int>(m,0)<<" "<<img_count.at<int>(m,0)<<endl;
        if (double(seg_count.at<int>(m,0))/double(img_count.at<int>(m,0))>=0.1)
        {
            labelMat.push_back(m);//记下所有区域内分割像素数占区域所有像素数大于一半的区域编号
            regioncount=regioncount+1;
        }
    }
    
    cout<<"满足条件的区域编号"<<labelMat<<endl;
    cout<<"区域个数"<<regioncount<<endl;
    
    /****求得每一个区域所有像素的maps值的平均值作为这个区域的maps值,同时也是预测数据******/
    Mat predictData1(0,1,CV_32FC1);//预测数据
    Mat predictDataMat1;
    for (m=0; m<maxid+1; m++)
    {
        int num=img_count.at<int>(m,0);
        Mat average_xmap(0,1,CV_32FC1),average_ymap(0,1,CV_32FC1),average_zmap(0,1,CV_32FC1),average_xzmap(0,1,CV_32FC1),average_yzmap(0,1,CV_32FC1);
        Scalar xmap_mean_value,ymap_mean_value,zmap_mean_value,xzmap_mean_value,yzmap_mean_value;
        for (i=0; i<num; i++)
        {
            average_xmap.push_back(float(xmap[pixelMat[m].at<int>(2*i+1)+1][pixelMat[m].at<int>(2*i)+1]));
            average_ymap.push_back(float(ymap[pixelMat[m].at<int>(2*i+1)+1][pixelMat[m].at<int>(2*i)+1]));
            average_zmap.push_back(float(zmap[pixelMat[m].at<int>(2*i+1)+1][pixelMat[m].at<int>(2*i)+1]));
            average_xzmap.push_back(float(xzmap[pixelMat[m].at<int>(2*i+1)+1][pixelMat[m].at<int>(2*i)+1]));
            average_yzmap.push_back(float(yzmap[pixelMat[m].at<int>(2*i+1)+1][pixelMat[m].at<int>(2*i)+1]));
        }
        xmap_mean_value=mean(average_xmap);
        ymap_mean_value=mean(average_ymap);
        zmap_mean_value=mean(average_zmap);
        xzmap_mean_value=mean(average_xzmap);
        yzmap_mean_value=mean(average_yzmap);
        //        cout<<xmap_mean_value<<" "<<ymap_mean_value<<" "<<zmap_mean_value<<" "<<xzmap_mean_value<<" "<<yzmap_mean_value<<endl;
        predictData1.push_back(float(xmap_mean_value.val[0]));
        predictData1.push_back(float(ymap_mean_value.val[0]));
        predictData1.push_back(float(zmap_mean_value.val[0]));
        predictData1.push_back(float(xzmap_mean_value.val[0]));
        predictData1.push_back(float(yzmap_mean_value.val[0]));
    }
    predictDataMat1=predictData1.reshape(0,maxid+1);//变为maxid+1行5列的矩阵
    cout<<"预测数据"<<predictDataMat1.size()<<endl;
    
    /*****根据labelMat里记载的序号从预测数据中挑出训练数据*******/
    
    for ( m = 0; m < regioncount; m++)
    {
        Scalar label_temp=labelMat.at<float>(m);
        //        cout<<label_temp.val[0]<<endl;
        trainingData1.push_back(predictDataMat1.at<float>(label_temp.val[0],0));
        trainingData1.push_back(predictDataMat1.at<float>(label_temp.val[0],1));
        trainingData1.push_back(predictDataMat1.at<float>(label_temp.val[0],2));
        trainingData1.push_back(predictDataMat1.at<float>(label_temp.val[0],3));
        trainingData1.push_back(predictDataMat1.at<float>(label_temp.val[0],4));
    }
    trainingDataMat1=trainingData1.reshape(0,regioncount);
    cout<<"正样本"<<trainingDataMat1.size()<<endl;
    
    Scalar kk=sum(regioncount);
    int k=kk.val[0];
    //    cout<<k<<endl;
    int N=sqrt(k*0.8/4);
    int M=N*N*4;
    Mat trainingData2(0,1,CV_32FC1);//负样本，取图像4个角像素
    Mat trainingDataMat2;
    int ai[4]={0,0,img_new1.rows-N,img_new1.rows-N},aj[4]={0,img_new1.cols-N,0,img_new1.cols-N};
    int bi[4]={N,N,img_new1.rows,img_new1.rows},bj[4]={N,img_new1.cols,N,img_new1.cols};
    
    for (m=0; m<4;m++)
    {
        for( i = ai[m]; i < bi[m]; i++)//原图image2比分割后图像小2个像素
        {
            for ( j = aj[m]; j < bj[m]; j++)
            {
                trainingData2.push_back(float(xmap[j+1][i+1]));
                trainingData2.push_back(float(ymap[j+1][i+1]));
                trainingData2.push_back(float(zmap[j+1][i+1]));
                trainingData2.push_back(float(xzmap[j+1][i+1]));
                trainingData2.push_back(float(yzmap[j+1][i+1]));
            }
        }
    }
    trainingDataMat2=trainingData2.reshape(0,M);
    cout<<"负样本"<<trainingDataMat2.size()<<endl;
    //    cout<<trainingDataMat2<<endl;
    
    for (i=0;i<M;i++)//正负样本合并
    {
        trainingDataMat1.push_back(trainingDataMat2.row(i));
    }
    cout<<"正负样本大小"<<trainingDataMat1.size()<<endl;
    
    int labels[k+M];//设置标记数组
    int val1=1,val2=-1;
    for (i=0;i<k;i++)
    {
        labels[i]=val1;
    }
    for (i=k;i<k+M;i++)
    {
        labels[i]=val2;
    }
    Mat labelsMat(k+M, 1, CV_32SC1, labels);
    cout<<labelsMat.size()<<endl;
    
    //设置SVM参数
    Ptr<SVM> svm = SVM::create();
    svm->setC(0.1);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
    
    //训练svm
    svm->train(trainingDataMat1,ROW_SAMPLE,labelsMat);
    
    Mat response;
    svm->predict(predictDataMat1,response);
    //    cout<<response<<endl;
    
    uchar white(255), black(0);//预测，每个区域统一标记
    for (m=0; m<maxid+1; m++)
    {
        int num1=img_count.at<int>(m,0);
        if (response.at<float>(m)==1)
        {
            for (i=0; i<num1; i++)
            {
                img_new1.at<float>(pixelMat[m].at<int>(2*i),pixelMat[m].at<int>(2*i+1))= white;
            }
        }
        else if (response.at<float>(m)==-1)
        {
            for (i=0; i<num1; i++)
            {
                img_new1.at<float>(pixelMat[m].at<int>(2*i),pixelMat[m].at<int>(2*i+1))= black;
            }
        }
    }
}


int main(int argc, char** argv)
{
    //********slic得到每个区域的标记*********//
    cv::Mat img, result;
    //    img = imread("/Users/qiuxinxin/temp/角毛藻显微图像/角毛藻图片/new/洛氏角毛藻/20081112-洛氏角毛藻-南海-BX61-40x-0000-24-0000-0000-0000-SCNUZYY-a01.tif");
    //    int numSuperpixel = atoi("6000");//设置superpixel的个数
    img = imread(argv[1]);
    int numSuperpixel = atoi(argv[3]);
    
    SLIC slic;
    slic.GenerateSuperpixels(img, numSuperpixel);
    int* label;
    int sz=img.rows*img.cols;
    label=slic.GetLabel();//得到每个区域的像素标记情况
    
    Mat img_new(Size(sz,1),CV_32FC1);
    
    int maxid=0;
    for (int k1=0;k1<sz;k1++)
    {
        img_new.at<float>(k1)=label[k1];
        if (label[k1]>maxid)
        maxid=label[k1];
        //        cout<<label[k1]<<endl;
        //        cout<<img_new.at<float>(k1)<<endl;
    }
    cout<<"maxid:"<<maxid<<endl;
    Mat img_new1=img_new.reshape(0,img.rows);//变成与原始图像一般大小
    cout<<"img_new1行列"<<img_new1.rows<<" "<<img_new1.cols<<endl;
    
    //*************gsdam得出5个maps*************//
    //    IplImage* image1=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/角毛藻图片/new/洛氏角毛藻/20081112-洛氏角毛藻-南海-BX61-40x-0000-24-0000-0000-0000-SCNUZYY-a01.tif",0);
    IplImage* image1=cvLoadImage(argv[1],0);
    IplImage* image_border = cvCreateImage( cvSize( image1 -> width+2, image1 -> height+2 ), IPL_DEPTH_8U, image1->nChannels);
    int height=image_border->height, width=image_border->width;
    double **xmap,**ymap,**zmap,**xzmap,**yzmap;
    xmap=new double*[width];
    ymap=new double*[width];
    zmap=new double*[width];
    xzmap=new double*[width];
    yzmap=new double*[width];
    for(int i=0; i<width; i++)
    {
        xmap[i]=new double[height];
        ymap[i]=new double[height];
        zmap[i]=new double[height];
        xzmap[i]=new double[height];
        yzmap[i]=new double[height];
    }
    
    gsdam(image1,img,xmap,ymap,zmap,xzmap,yzmap);
    
    //**********svm训练与预测************//
    //    Mat img_seg=imread("/Users/qiuxinxin/temp/角毛藻显微图像/Test/SvmSlicMethod/preprocess/xz_yz-maps-filter4/洛氏角毛藻/20081112-洛氏角毛藻-南海-BX61-40x-0000-24-0000-0000-0000-SCNUZYY-a01_xz_yz-filter.png",0);//resize后的分割图像
    Mat img_seg=imread(argv[2],0);//resize后的分割图像
    img_seg.convertTo(img_seg, CV_32FC1, 1.0/255.0);//从cv_8uc1转到cv_32fc1
    svm(img_seg,xmap,ymap,zmap,xzmap,yzmap,img_new1,maxid);
    
    //********保存结果*******//
    string add="-seg.png";
    save_img(img_new1,argv[1],add);
    
    for(int i=0; i<width; i++)
    {
        delete []xmap[i];
        delete []ymap[i];
        delete []zmap[i];
        delete []xzmap[i];
        delete []yzmap[i];
    }
    
    delete []xmap;
    delete []ymap;
    delete []zmap;
    delete []xzmap;
    delete []yzmap;
    return 0;
}