import os
import numpy as np
import tifffile

path='/home/ryuuyou/Project/self_net/data/care_liver/'
raw_data_path = path+'raw_data/'
train_data_path=path+'train_data/'

if not os.path.exists(train_data_path):
    os.mkdir(train_data_path)

xy_path = raw_data_path + 'xy/'
xy_lr_path = raw_data_path + 'xy_lr/'
xz_path = raw_data_path + 'xz/'

xy = []
xy_lr = []
xz = []

stride = 64
patch_size = 64

# signal_intensity_threshold=600  #parameter for selecting image patches containing signals
signal_intensity_threshold=0  #parameter for selecting image patches containing signals

xy_interval=4
xz_interval=12

for i in range(0, len(os.listdir(xy_path)), xy_interval):

    xy_img = tifffile.imread(xy_path + str(i + 1) + '.tif')
    xy_lr_img = tifffile.imread(xy_lr_path + str(i + 1) + '.tif')
    print(i + 1)

    for m in range(0, xy_img.shape[0] - patch_size + 1, stride):
        for n in range(0, xy_img.shape[1] - patch_size + 1, stride):
            crop_xy = xy_img[m:m + patch_size, n:n + patch_size]
            crop_xy_lr = xy_lr_img[m:m + patch_size, n:n + patch_size]

            if np.max(crop_xy >= signal_intensity_threshold):
                xy.append(crop_xy)
                xy_lr.append(crop_xy_lr)

for i in range(0, len(os.listdir(xz_path)), xz_interval):
    xz_img = tifffile.imread(xz_path + str(i + 1) + '.tif')
    print(i + 1)

    for m in range(0, xz_img.shape[0] - patch_size + 1, stride):
        for n in range(0, xz_img.shape[1] - patch_size + 1, stride):
            crop_xz = xz_img[m:m + patch_size, n:n + patch_size]

            if np.max(crop_xz >= signal_intensity_threshold):
                xz.append(crop_xz)

xy = np.array(xy, dtype=np.float32)
xy_lr = np.array(xy_lr, dtype=np.float32)
xz = np.array(xz, dtype=np.float32)
print(xy.shape, xy_lr.shape, xz.shape)

np.savez(path + '/train_data/train_data.npz', xy=xy, xy_lr=xy_lr, xz=xz)