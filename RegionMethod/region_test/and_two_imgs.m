clear all

folder_all1=dir('./result-xz');
falsefolder1=[];k1=1;
for i=1:numel(folder_all1)
    if folder_all1(i,1).name(1)=='.'
        falsefolder1(k1)=i;
        k1=k1+1;
    end  
end
folder_all1(falsefolder1)=[];

folder_all2=dir('./result-yz');
falsefolder2=[];k2=1;
for j=1:numel(folder_all2)
    if folder_all2(j,1).name(1)=='.'
        falsefolder2(k2)=j;
        k2=k2+1;
    end  
end
folder_all2(falsefolder2)=[];

nn=1;ratio=zeros(1,100);
for k=1:numel(folder_all1)
    outputfolder1=['./result-xz_yz-filter4','/',folder_all1(k,1).name];
    mkdir(outputfolder1);
    imgpath1=['./result-xz','/',folder_all1(k,1).name,'/','*.tif'];
    allimg1=dir(imgpath1);
    imgpath2=['./result-yz','/',folder_all2(k,1).name,'/','*.tif'];
    allimg2=dir(imgpath2);
    
    for m=1:numel(allimg1)
        bw1=imread(['./result-xz','/',folder_all1(k,1).name,'/',allimg1(m,1).name]);
        bw2=imread(['./result-yz','/',folder_all2(k,1).name,'/',allimg2(m,1).name]);
        [r,c]=size(bw1);
        bw=zeros(1,r*c);
        for n=1:r*c
            if (bw1(n)==255) && (bw2(n)==255)
                bw(n)=255;
            end
        end
        border=[1:r r*c-r+1:r*c 1:r:1+(c-1)*r r:r:r*c r+2:2*r-1 2+(c-2)*r:c*r-r-1 r+2:r:c*r-2*c+2*r 2*r-1:r:r*c-r-1];
        bw(border)=0;
        bw3=reshape(bw,r,c);
        
        l=bwlabel(bw3,8);
        s=regionprops(l,'Area');
        bw4=ismember(l,find([s.Area]>=max([s.Area])));
%         bw4=bwareaopen(bw3,30,8);
        s1=find(allimg1(m,1).name=='_');
        outputname=['./result-xz_yz-filter4','/',folder_all1(k,1).name,'/',...
        allimg1(m,1).name(1:s1(end)),'xz_yz-filter','.tif'];
        imwrite(bw4,outputname);
    end  
end