clear all;clc;

raw_path='D:\confocal_Thy1_neuron\raw_data\'; %folder path for the 3D image tiff

data_name='raw_data.tif';

raw_data=tiffread([raw_path data_name]); 
[m,n,p]=size(raw_data);

xy_path=[raw_path 'xy\'];
xy_lr_path=[raw_path 'xy_lr\'];
xz_path=[raw_path 'xz\'];
yz_path=[raw_path 'yz\'];

if ~exist(xy_path,'dir')
    mkdir(xy_path);
end

if ~exist(xy_lr_path,'dir')
    mkdir(xy_lr_path);
end

if ~exist(xz_path,'dir')
    mkdir(xz_path);
end

if ~exist(yz_path,'dir')
    mkdir(yz_path);
end


scale=0.21; % later pixel size / axial pixel size

for index=1:p
    image=squeeze(raw_data(:,:,index));
    
    imwrite(image,[raw_path 'xy\' int2str(index) '.tif']);
    downsample_img=imresize(image,[round(m*scale),n],'bicubic');
    downsample_img=imresize(downsample_img,[m,n],'bicubic');
    imwrite(downsample_img,[raw_path 'xy_lr\' int2str(index) '.tif']);
    disp(index);
end

xz_raw=reslice(raw_data,'xz',scale,1);

[m,n,p]=size(xz_raw);

for index=1:p
    image=squeeze(xz_raw(:,:,index));
    imwrite(image,[raw_path 'xz\' int2str(index) '.tif']);
    disp(index);
end

yz_raw=reslice(raw_data,'yz',scale,1);

[m,n,p]=size(yz_raw);

for index=1:p
    image=squeeze(yz_raw(:,:,index));
    imwrite(image,[raw_path 'yz\' int2str(index) '.tif']);
    disp(index);
end


     
     

    
    

