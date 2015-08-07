clear all

inputdir='/Users/qiuxinxin/temp/角毛藻显微图像/Test/RegionMethod/三步对比/Results-filter5';
outputdir='./result';
folder_all=dir(inputdir);

falsefolder=[];k1=1;
for i=1:numel(folder_all)
    if folder_all(i,1).name(1)=='.'
        falsefolder(k1)=i;
        k1=k1+1;
    end  
end

folder_all(falsefolder)=[];

for i=1:numel(folder_all)
    imgpath=[inputdir,'/',folder_all(i,1).name,'/','*.tif'];
    allimg=dir(imgpath);
    outputfolder1=[outputdir,'/',folder_all(i,1).name];
    mkdir(outputfolder1);
    falsefile1=[];k2=1;
    for j=1:numel(allimg)
        if allimg(j,1).name(1)=='.' || allimg(j,1).isdir==1
            falsefile1(k2)=j;
            k2=k2+1;
        end
    end
    allimg(falsefile1)=[];
    
    for j=1:numel(allimg)
        s1=find(allimg(j,1).name=='-');
        outputname1=[outputfolder1,'/',allimg(j,1).name(1:s1(1)),'ske','.tif'];
        img=imread([inputdir,'/',folder_all(i,1).name,'/',allimg(j,1).name]);
%         bw=bwmorph(skeleton(img)>35,'skel',Inf);
        bw=bwmorph(img,'dilate',1);
        bw=bwmorph(bw,'majority',Inf);
%         bw=bwmorph(bw,'open',2);
        [bw,rad]=skeleton(bw);
        bw=bwmorph(bw,'majority',Inf);
        bw=bwmorph(skeleton(bw)>35,'skel',Inf);
        imwrite(bw,outputname1);
    end
end

