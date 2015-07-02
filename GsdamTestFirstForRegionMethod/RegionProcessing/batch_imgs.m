clear all

folder_all=dir('./Results-map');

falsefolder=[];k1=1;
for i=1:numel(folder_all)
    if folder_all(i,1).name(1)=='.'
        falsefolder(k1)=i;
        k1=k1+1;
    end  
end

folder_all(falsefolder)=[];

for j=1:numel(folder_all)
    imgpath=['./Results-map','/',folder_all(j,1).name,'/','*.tif'];
    allimg=dir(imgpath);
    outputfolder=['./result','/',folder_all(j,1).name];
    mkdir(outputfolder);
    
    falsefile1=[];k2=1;
    for i=1:numel(allimg)
        if allimg(i,1).name(1)=='.' || allimg(i,1).isdir==1
            falsefile1(k2)=i;
            k2=k2+1;
        end
    end
    allimg(falsefile1)=[];
    
    eps=0.000001;
    IterMax=20;lambda=0.025;
    for i=1:numel(allimg)
        s1=find(allimg(i,1).name=='-');
        outputname=['./result','/',folder_all(j,1).name,'/',allimg(i,1).name(1:s1),'region','.tif'];
        u0=imread(['./Results-map','/',folder_all(j,1).name,'/',allimg(i,1).name]);
%         h=imhist(u0);
%         bar(h);
        u1=otsu(u0,2);
%         imagesc(u1),axis image off,colormap(gray)
%         imwrite(mat2gray(u1),'./result/1.tif')
%         level = graythresh(u0);
%         level = multithresh(u0,2);
%         u1 = imquantize(u0,level);
%         imwrite(mat2gray(u1),outputname);
        u1=im2bw(uint8(u1),0.004);
        l=bwlabel(u1,8);
        s=regionprops(l,'Area');
        bw2=ismember(l,find([s.Area]>=200));
%         u2=im2bw(uint8(u1),0.008);
%         l=bwlabel(u2,8);
%         s=regionprops(l,'Area');
%         bw2=ismember(l,find([s.Area]>=150));
%         imwrite(bw2,outputname);
            imwrite(mat2gray(bw2),outputname);
    end
end