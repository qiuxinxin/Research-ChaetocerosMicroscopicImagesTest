clear all

folder_all=dir('./maps_new2');

falsefolder=[];k1=1;
for i=1:numel(folder_all)
    if folder_all(i,1).name(1)=='.'
        falsefolder(k1)=i;
        k1=k1+1;
    end  
end

folder_all(falsefolder)=[];

for j=1:numel(folder_all)
    imgpath=['./maps_new2','/',folder_all(j,1).name,'/','*.tif'];
    allimg=dir(imgpath);
    outputfolder1=['./result-xz','/',folder_all(j,1).name];
    mkdir(outputfolder1);
    outputfolder2=['./result-yz','/',folder_all(j,1).name];
    mkdir(outputfolder2);
    
    falsefile1=[];k2=1;
    for i=1:numel(allimg)
        if allimg(i,1).name(1)=='.' || allimg(i,1).isdir==1
            falsefile1(k2)=i;
            k2=k2+1;
        end
    end
    allimg(falsefile1)=[];
    
    for i=1:numel(allimg)
        s1=find(allimg(i,1).name=='-');
        outputname1=['./result-xz','/',folder_all(j,1).name,'/',allimg(i,1).name(1:s1(end)),'region','.tif'];
        outputname2=['./result-yz','/',folder_all(j,1).name,'/',allimg(i,1).name(1:s1(end)),'region','.tif'];
        u0=imread(['./maps_new2','/',folder_all(j,1).name,'/',allimg(i,1).name]);
        u1=otsu(u0,2);
        u1=im2bw(uint8(u1),0.004);
        [r,c]=size(u1);
        border=[1:r r*c-r+1:r*c 1:r:1+(c-1)*r r:r:r*c r+2:2*r-1 2+(c-2)*r:c*r-r-1 r+2:r:c*r-2*c+2*r 2*r-1:r:r*c-r-1];
        u1(border)=0;
        l=bwlabel(u1,8);
        s=regionprops(l,'Area');
%         bw2=ismember(l,find([s.Area]>=200));
        bw2=ismember(l,find([s.Area]>=max([s.Area])));
        if mod(i,2)~=0
        imwrite(mat2gray(bw2),outputname1);
        else
            imwrite(mat2gray(bw2),outputname2);
        end
    end
end