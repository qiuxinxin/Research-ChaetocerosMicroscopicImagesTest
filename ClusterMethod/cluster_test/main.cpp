#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>


using namespace cv;
using namespace std;
typedef IplImage* IPL;

extern double compute_direction_angles(double,double,double,double,double *,double *,double *,double *);
extern IPL save_img(IPL,char *,string);


int main(int argc, char* argv[])
{
    IPL src1=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/cluster/cluster_test/洛氏角毛藻40-1.tif",0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像,1为RGB图像
    IPL src2=cvLoadImage("/Users/qiuxinxin/temp/角毛藻显微图像/cluster/cluster_test/洛氏角毛藻40-1.tif",1);
    IPL src = cvCreateImage( cvSize( src1 -> width+2, src1 -> height+2 ), IPL_DEPTH_8U, src1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称
    IPL srcc = cvCreateImage( cvSize( src2 -> width+2, src2 -> height+2 ),IPL_DEPTH_8U, src2->nChannels);
    IPL image_x=cvCloneImage(src);
    IPL image_y=cvCloneImage(src);
    IPL image_z=cvCloneImage(src);
    IPL image_xy=cvCloneImage(src);
    IPL image_xz=cvCloneImage(src);
    IPL image_yz=cvCloneImage(src);
    
    int height=src->height, width=src->width;
    //        printf("h=%d w=%d\n",height,width);
    double **c1;
    double **c2;
    double **c3;
    double **c4;
    double **xmap;
    double **ymap;
    double **zmap;
    double **xymap;
    double **xzmap;
    double **yzmap;
    
    c1 = new double*[width];//x方向角的角度 new表示动态内存分配 c1作为一个指向指针的指针，它指向一个包含width个元素的指针数组
    c2 = new double*[width];
    c3 = new double*[width];
    c4 = new double*[width];
    xmap=new double*[width];
    ymap=new double*[width];
    zmap=new double*[width];
    xymap=new double*[width];
    xzmap=new double*[width];
    yzmap=new double*[width];
    
    
    cvCopyMakeBorder(src1, src, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
    cvCopyMakeBorder(src2, srcc, cvPoint(1,1), IPL_BORDER_CONSTANT);
    
    for(int i=0; i<width; i++)
    {
        c1[i] = new double[height];//每个指针元素指向一个有height个元素的数组，这样就构建了有width行height列的数组
        c2[i] = new double[height];
        c3[i] = new double[height];
        c4[i] = new double[height];
        xmap[i]=new double[height];
        ymap[i]=new double[height];
        zmap[i]=new double[height];
        xymap[i]=new double[height];
        xzmap[i]=new double[height];
        yzmap[i]=new double[height];
        
    }
    
    int x,y;
    CvScalar Ix0y0,Ix0y1,Ix1y1,Ix1y0;//cvscalar的结构为一个double型的4成员的数组，它提供了四个成员，最多就可对应四个通道
    double w1,w2,w3,* w1_z,* w2_z,* w3_z,dis,* dis_z;
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            Ix0y0=cvGet2D(src,y,x);//提取图像(y,x)的像素值，与通常坐标系相反
            //            printf("y=%d x=%d ixoyo=%f\n",y,x,Ix0y0.val[0]);
            Ix0y1=cvGet2D(src,(y+1),x);
            Ix1y1=cvGet2D(src,(y+1),(x+1));
            //            printf("y=%d x=%d ixoyo=%f\n",y+1,x+1,Ix1y1.val[0]);
            Ix1y0=cvGet2D(src,y,(x+1));
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
            //            printf("x=%f\n",xmap[x][y]);
            cvSet2D(image_x,y,x,cvRealScalar(xmap[x][y]));//将map的值赋给图像中每个像素点
            cvSet2D(image_y,y,x,cvRealScalar(ymap[x][y]));//cvrealscalar把数组的第一个值置为想要的数（即为 double val0 ），其他三个数均为零。
            cvSet2D(image_z,y,x,cvRealScalar(zmap[x][y]));
            
        }
    }
    
    
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            xymap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+ymap[x][y]*ymap[x][y]);
            xzmap[x][y]=sqrt(xmap[x][y]*xmap[x][y]+zmap[x][y]*zmap[x][y]);
            yzmap[x][y]=sqrt(ymap[x][y]*ymap[x][y]+zmap[x][y]*zmap[x][y]);
            cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
        }
    }
    //
    double xymap_max=xymap[0][0];
    double xymap_min=xymap[0][0];
    double xzmap_max=xzmap[0][0];
    double xzmap_min=xzmap[0][0];
    double yzmap_max=yzmap[0][0];
    double yzmap_min=yzmap[0][0];
    for(y=0; y<height-1; y++)
    {
        for(x=0; x<width-1; x++)
        {
            if (xymap[x][y]>xymap_max)
            {
                xymap_max=xymap[x][y];
            }
            else {}
            if (xymap[x][y]<xymap_min)
            {
                xymap_min=xymap[x][y];
            }
            else {}
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
            xymap[x][y]=255*( (xymap[x][y]-xymap_min)/(xymap_max-xymap_min));
            //               printf("xy=%f\n",xymap[x][y]);
            xzmap[x][y]=255*( (xzmap[x][y]-xzmap_min)/(xzmap_max-xzmap_min));//对xzmap,yzmap进行255归一化
            yzmap[x][y]=255*( (yzmap[x][y]-yzmap_min)/(yzmap_max-yzmap_min));
            //                printf("y=%d x=%d xxx=%f yyy=%f zzz=%f xxy=%f xxz=%f xyz=%f\n",y,x,xmap[x][y],ymap[x][y],zmap[x][y],xymap[x][y],xzmap[x][y],yzmap[x][y]);
            cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
            cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
            cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
        }
    }
    
    for(int i=0; i<width; i++)
    {
        delete []c1[i];//先撤销指针元素所指向的数组
        delete []c2[i];
        delete []c3[i];
        delete []c4[i];
        delete []xmap[i];
        delete []ymap[i];
        delete []zmap[i];
        delete []xymap[i];
        delete []xzmap[i];
        delete []yzmap[i];
    }
    delete []c1;//在销毁的过程，先销毁指针数组每个元素指向的数组，然后再销毁这个指针数组
    delete []c2;
    delete []c3;
    delete []c4;
    delete []xmap;
    delete []ymap;
    delete []zmap;
    delete []xymap;
    delete []xzmap;
    delete []yzmap;
    
    int i,j;
    
    Mat imagex=cvarrToMat(image_x);//iplimage转mat
    Mat imagey=cvarrToMat(image_y);
    Mat imagez=cvarrToMat(image_z);
    Mat imagexz=cvarrToMat(image_xz);
    Mat imageyz=cvarrToMat(image_yz);
    //cout<<imageyz<<endl;
    Mat image1=cvarrToMat(src);
    Mat image2=cvarrToMat(srcc);
    Mat image3=cvarrToMat(src2);
    
    Mat sample=Mat((height-2)*(width-2),2,CV_32FC1);//生成height*width行5列矩阵，注意采样点格式为32bit浮点数
    Mat labels1((height-2)*(width-2), 1, CV_32SC1);
    Mat img1(image3.size(), CV_8UC3);
    
    cout<<sample<<endl;
    uchar*p1,*p2,*p3,*p4,*p5,*p;
    for(i=0; i < height-2; i++)
    {
        p1 = imagex.ptr<uchar>(i+1);
        p2 = imagey.ptr<uchar>(i+1);
        p3 = imagez.ptr<uchar>(i+1);
        p4 = imagexz.ptr<uchar>(i+1);
        p5 = imageyz.ptr<uchar>(i+1);
        p = image2.ptr<uchar>(i+1);
        // cout<<p<<endl;
        for(j=0; j< width-2; j++)
        {
            //            sample.at<float>(i*(width-2)+j,0)= float(p[3*(j+1)]);
            //            sample.at<float>(i*(width-2)+j,1)= float(p[3*(j+1)+1]);
            //            sample.at<float>(i*(width-2)+j,2)= float(p[3*(j+1)+2]);
            //            sample.at<float>(i*(width-2)+j,0)= float(p1[j+1]);
            //            sample.at<float>(i*(width-2)+j,1)= float(p2[j+1]);
            //            sample.at<float>(i*(width-2)+j,2)= float(p3[j+1]);
            //
            sample.at<float>(i*(width-2)+j,0)= float(p4[j+1]);
            sample.at<float>(i*(width-2)+j,1)= float(p5[j+1]);
            
            //            cout<<sample.at<float>(i*(width-2)+j,0)<<endl;
        }
    }
    
    //归一化
    Mat samp1=Mat((height-2)*(width-2),2,CV_32FC1);
    for (i=0;i<2;i++)
    {
        double min=1, max=1;
        Mat samp=sample.colRange(i,i+1).clone();//提取sample的每一列特征
        minMaxIdx(samp,&min,&max);
        for (j=0;j<sample.rows;j++)
        {
            samp1.at<float>(j,i)=(samp.at<float>(j,i)-min)/(max-min);
        }
    }
    
    int cluster_num=2;
    Mat centers1(cluster_num, 1, sample.type());//用来存储聚类后的中心点
    kmeans(samp1, cluster_num, labels1, TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 10, 1.0), 1,KMEANS_PP_CENTERS, centers1);//TermCriteria控制迭代算法的终止条件
    
    int m=0;
    for( i = 0; i < image3.rows; i++ )
        for( j = 0; j < image3.cols; j++ )
        {
            int index = labels1.at<int>(m,0);
            //printf("in=%d\n",index);
            m++;
            if( index == -1 )
                img1.at<Vec3b>(i,j) = Vec3b(255,0,0);
            else if (index < 0 || index > cluster_num)
                img1.at<Vec3b>(i,j) = Vec3b(0,255,0);
            else if (index==0)
                img1.at<Vec3b>(i,j) = Vec3b(255,255,255);
            else
                img1.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    string savingfile(argv[1]);
    string shortstring=savingfile.substr(savingfile.find_last_of("/")+1,savingfile.length()-savingfile.find_last_of("/"));//只提取图片名字，不带路径
    shortstring.erase(shortstring.find_last_of("."));//去除文件扩展名
    string add="_kmeans++xz_yz.tif";
    shortstring+=add;//加入新的标示及扩展名
    imwrite(shortstring,img1);
    
    //            namedWindow("K-means++");
    //            imshow("K-means++", img1);
    //            waitKey();
    
    cvReleaseImage(&image_x);
    cvReleaseImage(&image_y);
    cvReleaseImage(&image_z);
    cvReleaseImage(&image_xz);
    cvReleaseImage(&image_yz);
    
    return 0;
}