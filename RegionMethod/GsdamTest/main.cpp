#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <string.h>
#include <string>

using namespace std;
typedef IplImage* IPL;

extern double compute_direction_angles(double,double,double,double,double *,double *,double *,double *);
extern IPL map_binarization(IplImage*,IPL);
extern IPL and_two_binary_imgs(IPL,IPL);
extern IPL extract_max_area_contour(IPL);
extern IPL and_template_original_imgs(IPL,IPL);
extern IPL save(IPL,char *,int);
extern IPL save_img(IPL,char *,string);
extern int check(char*);
extern int model(char*);


int main(int argc, char* argv[])
{
    IPL src1=cvLoadImage(argv[1],0);//将图像文件加载至内存，自动分配图像数据结构所需的内存，该函数执行完后将返回一个指针，0代表图像为灰度图像
    IPL src = cvCreateImage( cvSize( src1 -> width+2, src1 -> height+2 ), IPL_DEPTH_8U, src1->nChannels);//创造一个图像header并分配给图像数据（图像宽高，图像元素的比特数，这里为8，每像素的通道数），->可以看到图像所有的性质 argv为输入各参数的名称
    IPL src2=cvLoadImage(argv[1],1);//1代表图像load为3通道图像
    IPL image_x=cvCloneImage(src);
    IPL image_y=cvCloneImage(src);
    IPL image_z=cvCloneImage(src);
    IPL image_xy=cvCloneImage(src);
    IPL image_xz=cvCloneImage(src);
    IPL image_yz=cvCloneImage(src);
    IPL image_xz_BF=cvCloneImage(src);
    IPL image_yz_BF=cvCloneImage(src);
//    IPL image_x_BF=cvCloneImage(src);
//    IPL image_y_BF=cvCloneImage(src);
    
    int height=src->height, width=src->width;
//    printf("h=%d w=%d\n",height,width);
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
    double **Ix;
    double **Iy;
    
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
    Ix=new double*[width];
    Iy=new double*[width];
    
    
    cvCopyMakeBorder(src1, src, cvPoint(1,1), IPL_BORDER_CONSTANT);//复制图像并且制作边界，Bordertype=IPL_BORDER_CONSTANT时，有一个像素宽的黑色边界（为了后面计算原图像的边界点方便），指定（1，1）为原点坐标，拷贝图像，因此输出图像要对应扩大
    
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
//        Ix[i]=new double[height];
//        Iy[i]=new double[height];
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
//            printf("x=%f y=%f z=%f\n",xmap[x][y],ymap[x][y],zmap[x][y]);
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
//                printf("xy=%f\n",xymap[x][y]);
                xzmap[x][y]=255*( (xzmap[x][y]-xzmap_min)/(xzmap_max-xzmap_min));//对xzmap,yzmap进行255归一化
                yzmap[x][y]=255*( (yzmap[x][y]-yzmap_min)/(yzmap_max-yzmap_min));
//                printf("y=%d x=%d xxx=%f yyy=%f zzz=%f xxy=%f xxz=%f xyz=%f\n",y,x,xmap[x][y],ymap[x][y],zmap[x][y],xymap[x][y],xzmap[x][y],yzmap[x][y]);
                cvSet2D(image_xy,y,x,cvRealScalar(xymap[x][y]));
                cvSet2D(image_xz,y,x,cvRealScalar(xzmap[x][y]));
                cvSet2D(image_yz,y,x,cvRealScalar(yzmap[x][y]));
            }
        }
    
//    double xx, yy, zz, xy, xz, yz;
//    xx=0;yy=0;zz=0;xy=0;xz=0;yz=0;
//    for(y=0; y<height-2; y++)
//    {
//        for(x=0; x<width-2; x++)
//        {
//            xx=xmap[x][y]+xx;
//            yy=ymap[x][y]+yy;
//            zz=zmap[x][y]+zz;
//            xy=xymap[x][y]+xy;
//            xz=xzmap[x][y]+xz;
//            yz=yzmap[x][y]+yz;
//        }
//    }
//    printf("ax=%f ay=%f az=%f axy=%f axz=%f ayz=%f\n",xx/((height-2)*(width-2)),yy/((height-2)*(width-2)),zz/((height-2)*(width-2)),xy/((height-2)*(width-2)),xz/((height-2)*(width-2)),yz/((height-2)*(width-2)));
//    
//    string name1="_x-map.tif";
//    save_img(image_x,argv[1],name1);
//    string name2="_y-map.tif";
//    save_img(image_y,argv[1],name2);
//    string name3="_z-map.tif";
//    save_img(image_z,argv[1],name3);
//    string name4="_xy-map.tif";
//    save_img(image_xy,argv[1],name4);
    string name5="_1_xz-map.tif";
    save_img(image_xz,argv[1],name5);
    string name6="_2_yz-map.tif";
    save_img(image_yz,argv[1],name6);
//
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

    
    cvSmooth(image_xz,image_xz_BF,CV_BILATERAL,-1,1,40,20);//双向滤波(bilateral filtering)，image_xz_BF为输出图像
    cvSmooth(image_yz,image_yz_BF,CV_BILATERAL,-1,1,40,20);
    string name1="_3_xz_bf.tif";
    save_img(image_xz_BF,argv[1],name1);
    string name2="_4_yz_bf.tif";
    save_img(image_yz_BF,argv[1],name2);
    
    IPL image_xz_BF_binary=cvCreateImage(cvGetSize(image_xz),IPL_DEPTH_8U,1);
    cvZero(image_xz_BF_binary);//初始化图片，值都为0
    image_xz_BF_binary=map_binarization(image_xz_BF,image_xz_BF_binary);//大津法二值化map图像
    
    IPL image_yz_BF_binary=cvCreateImage(cvGetSize(image_yz),IPL_DEPTH_8U,1);
    cvZero(image_yz_BF_binary);
    image_yz_BF_binary=map_binarization(image_yz_BF,image_yz_BF_binary);
    
    string name3="_5_xz_bfb.tif";
    save_img(image_xz_BF_binary,argv[1],name3);
    string name4="_6_yz_bfb.tif";
    save_img(image_yz_BF_binary,argv[1],name4);
    
    IPL image_xz_and_yz=and_two_binary_imgs(image_xz_BF_binary,image_yz_BF_binary);//与操作
    string name7="_7_xz_yz.tif";
    save_img(image_xz_and_yz,argv[1],name7);
    
    IPL dilate_src=cvCloneImage(image_xz_and_yz);
    IPL imagesmoothM=cvCloneImage(dilate_src);
    IPL imagesmoothM2=cvCloneImage(dilate_src);
    
    cvSmooth(dilate_src,imagesmoothM,CV_MEDIAN,5,5);//中值滤波
    cvSmooth(imagesmoothM,imagesmoothM2,CV_MEDIAN,3,3);
    string name8="_8_smooth.tif";
    save_img(imagesmoothM2,argv[1],name8);
    
    IPL image_dilate=cvCloneImage(dilate_src);
    IPL image_open=cvCloneImage(dilate_src);
    IPL image_close=cvCloneImage(dilate_src);
    IPL image_temp=cvCloneImage(dilate_src);
    cvSetZero(image_temp);
    
    IplConvKernel *element=0;
    IplConvKernel *element2=0;
    
    element=cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_ELLIPSE);//创建结构元素，定义了下面操作所需的模板，前两个参数定义模板大小，然后为参考点的坐标，模板的类型（椭圆）
    element2=cvCreateStructuringElementEx(2,2,0,0,CV_SHAPE_ELLIPSE);
    
    cvMorphologyEx(imagesmoothM2,image_close,image_temp,element,CV_MOP_CLOSE,5);//imagesmoothM2进行形态学闭操作，temp为临时变量，需要与原数组大小相同，element为自定义模板，5为迭代次数
    cvDilate(image_close,image_dilate,element2,5);//膨胀操作
    string name9="_9_close.tif";
    save_img(image_close,argv[1],name9);
    string name10="_10_dilate.tif";
    save_img(image_dilate,argv[1],name10);
    
    IPL image_contour=extract_max_area_contour(imagesmoothM2);//提取出最大的轮廓
    string name11="_11_contour.tif";
    save_img(image_contour,argv[1],name11);
    
    cvErode(image_contour,image_contour,element2,5);//最大轮廓进行腐蚀操作
    cvMorphologyEx(image_contour,image_open,image_temp,element,CV_MOP_OPEN,3);//最大轮廓进行腐蚀操作后进行形态学开操作
    string name12="_12_open.tif";
    save_img(image_open,argv[1],name12);
    
    IPL  image_final=and_template_original_imgs(src2,image_open);
    string name17="_13_final.tif";
    save_img(image_final,argv[1],name17);

//    int flag;
//    if(argc==2)//判断参数个数
//    {
//        flag=0;
//        save(image_final,argv[1],flag);
//    }
//    else
//    {
//        flag =1;
//        for(int n=2;n<argc;n+=2)
//        {
//            if(check(argv[n]))//argv个数与argc相同，检查满足参数要求(-x,-g)
//            {
//                if(check(argv[n+1]))//检查输出图像格式是否正确（tif jpg）
//                {
//                    switch(model(argv[n]))
//                    {
//                        case 0:save(src1,argv[n+1],flag);break;
//                        case 1:save(image_x,argv[n+1],flag);break;
//                        case 2:save(image_y,argv[n+1],flag);break;
//                        case 3:save(image_z,argv[n+1],flag);break;
//                        case 4:save(image_xz,argv[n+1],flag);break;
//                        case 5:save(image_yz,argv[n+1],flag);break;
//                        case 6:save(image_open,argv[n+1],flag);break;
//                        case 7:save(image_close,argv[n+1],flag);break;
//                        case 8:save(image_contour,argv[n+1],flag);break;
//                        case 9:save(image_final,argv[n+1],flag);break;
//                        default:break;
//                    }
//                }
//                else
//                    cout << argv[n+1]  <<":   "<<"The Wrong Format" <<endl;
//            }
//            else
//                cout << argv[n]  <<":   " <<"Error Option" <<endl;
//        }
//    }
    
    cvReleaseImage(&image_x);
    cvReleaseImage(&image_y);
    cvReleaseImage(&image_z);
    cvReleaseImage(&image_dilate);
    cvReleaseImage(&image_xz_BF_binary);
    cvReleaseImage(&image_yz_BF_binary);
    cvReleaseImage(&image_xz_and_yz);
    cvReleaseImage(&imagesmoothM2);
    cvReleaseImage(&image_close);
    cvReleaseImage(&image_contour);
    cvReleaseImage(&image_open);
    cvReleaseImage(&image_final);
    
    return 0;
}

