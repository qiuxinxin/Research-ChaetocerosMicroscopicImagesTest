%%将rgb图像转化为二值图像
clear all

imgpath=['./dataset-process/','*.tif'];
allimg=dir(imgpath);
outputfolder1=['./binarization_img'];
mkdir(outputfolder1);

falsefile1=[];k2=1;
for i=1:numel(allimg)
    if allimg(i,1).name(1)=='.' || allimg(i,1).isdir==1
        falsefile1(k2)=i;
        k2=k2+1;
    end
end
allimg(falsefile1)=[];

for i=1:numel(allimg)
    bw=imread(['./dataset-process/',allimg(i,1).name]);
    
    [r,c,channel]=size(bw);
    for k=1:r %如果是蓝绿两色，转化为黑白两色
        for m=1:c
            if bw(k,m,1)==255 && bw(k,m,2)==255 && bw(k,m,3)==255
            else
                bw(k,m,1)=0;
                bw(k,m,2)=0;
                bw(k,m,3)=0;
            end
        end
    end
    bw2=rgb2gray(bw);
    s=find(allimg(i,1).name=='.');
    outputname1=[outputfolder1,'/',allimg(i,1).name(1:s-1),'-bi','.tif'];
    imwrite(bw2,outputname1);
end
