%%对训练后分类图像进行处理，得到最终的分割图像
clear all

folder_all=dir('/Users/qiuxinxin/temp/角毛藻显微图像/Test/预分割影响svm结果实验/Results');

falsefolder=[];k1=1;
for i=1:numel(folder_all)
    if folder_all(i,1).name(1)=='.'
        falsefolder(k1)=i;
        k1=k1+1;
    end  
end
folder_all(falsefolder)=[];

for j=1:numel(folder_all)
    imgpath=['/Users/qiuxinxin/temp/角毛藻显微图像/Test/预分割影响svm结果实验/Results','/',folder_all(j,1).name,'/','*.tif'];
    allimg=dir(imgpath);
    outputfolder1=['./Results-filter','/',folder_all(j,1).name];
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
        bw1=imread(['/Users/qiuxinxin/temp/角毛藻显微图像/Test/预分割影响svm结果实验/Results','/',folder_all(j,1).name,'/',allimg(i,1).name]);
        %     [r,c,channel]=size(bw1);
        %     for k=1:r %如果是蓝绿两色，转化为黑白两色
        %         for m=1:c
        %             if bw1(k,m,1)==0 && bw1(k,m,2)==255 && bw1(k,m,3)==0
        %                 bw1(k,m,1)=255;
        %                 bw1(k,m,2)=255;
        %                 bw1(k,m,3)=255;
        %             end
        %             if bw1(k,m,1)==0 && bw1(k,m,2)==0 && bw1(k,m,3)==255
        %                 bw1(k,m,1)=0;
        %                 bw1(k,m,2)=0;
        %                 bw1(k,m,3)=0;
        %             end
        %         end
        %     end
        
        bw2=rgb2gray(bw1);
        [r,c]=size(bw2);
        border=[1:r r*c-r+1:r*c 1:r:1+(c-1)*r r:r:r*c r+2:2*r-1 2+(c-2)*r:c*r-r-1 r+2:r:c*r-2*c+2*r 2*r-1:r:r*c-r-1];
        bw2(border)=0;
%         for k=1:10
%             bw3=bwareaopen(bw2,50+20*(k-1),8);
%             l=bwlabel(bw3,8);
%             s=regionprops(l,'Area');
%             if numel(s)<=4
%                 break;
%             end
%         end
        bw3=bwareaopen(bw2,35,8);
        l=bwlabel(bw3,8);
        s=regionprops(l,'Area');
        if numel(s)>5
            area=sort([s.Area],'descend');
            marea=area(5);
            bw3=ismember(l,find([s.Area]>=marea));
        end
        s=find(allimg(i,1).name=='.');
        outputname1=[outputfolder1,'/',allimg(i,1).name(1:s-1),'-filter','.tif'];
        imwrite(bw3,outputname1);
    end   
end