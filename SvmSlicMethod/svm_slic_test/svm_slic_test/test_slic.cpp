
#include "slic.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

using namespace std;

extern Mat* save_img(Mat,char *,string);

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
////////
void gsdam(IplImage* image1, Mat image2, double **xmap,double **ymap,double **zmap, double **xzmap, double **yzmap)
{
    IplImage* image_border = cvCreateImage( cvSize( image1 -> width+2, image1 -> height+2 ), IPL_DEPTH_8U, image1->nChannels);
    
    int height=image_border->height, width=image_border->width;
    double **c1;
    double **c2;
    double **c3;
    double **c4;
//    double **xmap;
//    double **ymap;
//    double **zmap;
//    double **xzmap;
//    double **yzmap;
    
    c1 = new double*[width];
    c2 = new double*[width];
    c3 = new double*[width];
    c4 = new double*[width];
//    xmap=new double*[width];
//    ymap=new double*[width];
//    zmap=new double*[width];
//    xzmap=new double*[width];
//    yzmap=new double*[width];
    
    
    cvCopyMakeBorder(image1, image_border, cvPoint(1,1),IPL_BORDER_CONSTANT);
    
    for(int i=0; i<width; i++)
    {
        c1[i] = new double[height];
        c2[i] = new double[height];
        c3[i] = new double[height];
        c4[i] = new double[height];
//        xmap[i]=new double[height];
//        ymap[i]=new double[height];
//        zmap[i]=new double[height];
//        xzmap[i]=new double[height];
//        yzmap[i]=new double[height];
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

int main(int argc, char** argv)
{

	cv::Mat img, result;
	
    img = imread("/Users/qiuxinxin/Downloads/SLIC-superpixel-master/bird_color.jpg");
//	img = imread(argv[1]);
//	int numSuperpixel = atoi(argv[2]);
    int numSuperpixel = atoi("200");
	SLIC slic;
	slic.GenerateSuperpixels(img, numSuperpixel);
    int* label;
    int sz=img.rows*img.cols;
    label=slic.GetLabel();
    Mat img_new(Size(img.cols*img.rows,1),CV_8UC1);
    for (int k=0;k<sz;k++)
    {
        img_new.at<uchar>(k)=label[k];
//        cout<<label[k]<<endl;
        
    }
//    imshow("img",img_new);
//    cout<<img_new<<endl;
    double minid,maxid;
    minMaxLoc(img_new, &minid, &maxid);
//    cout<<maxid<<endl;

    //*************gsdam得出5个maps*************//
    IplImage* image1=cvLoadImage("/Users/qiuxinxin/Downloads/SLIC-superpixel-master/bird_color.jpg",0);
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
    
//    for(int y=0; y<height-1; y++)
//    {
//        for(int x=0; x<width-1; x++)
//        {
//            cout<<xmap[x][y]<<endl;
//        }
//    }
 //****************************//
    int x,y;
    Mat id,diff;
    img_new.copyTo(id);
    for (int m=0; m<maxid+1; m++)
    {
        Mat average_xmap;
        for (int k=0; k<sz; k++)
        {
            if (id.at<uchar>(k)!=maxid+1 && img_new.at<uchar>(k)==m)
            {
                average_xmap.push_back(xmap[k]);
//                cout<<float(xmap[k][1])<<endl;
//                cout<<average_xmap<<endl;
            }
        }
    }
    
    
    
	cv::imwrite("result.jpg", result);
//    string add="-slic.tif";
//    save_img(result,argv[1],add);
}