
clear all
path=['/Users/qiuxinxin/temp/角毛藻显微图像/Test/RegionMethod/Results-svm训练后4'];
folder_all=dir(path);

falsefolder=[];k1=1;
for i=1:numel(folder_all)
    if folder_all(i,1).name(1)=='.'
        falsefolder(k1)=i;
        k1=k1+1;
    end  
end
folder_all(falsefolder)=[];

for j=1:numel(folder_all)
    imgpath=[path,'/',folder_all(j,1).name,'/','*.tif'];
    allimg=dir(imgpath);
    outputfolder1=['./Results-filter41','/',folder_all(j,1).name];
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
        bw1=imread([path,'/',folder_all(j,1).name,'/',allimg(i,1).name]);
        
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
        
        %         l=bwlabel(bw2,8);
        %         s=regionprops(l,'Area');
        %         bw3=ismember(l,find([s.Area]>=max([s.Area])));
        bw3=bwareaopen(bw2,300,8);
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