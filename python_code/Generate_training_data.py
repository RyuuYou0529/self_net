import os
import numpy as np
import tifffile

path='/home/ryuuyou/Project/self_net/data/visor'
raw_data_path = os.path.join(path, 'raw_data/')
train_data_path=os.path.join(path, 'train_data/')

if not os.path.exists(train_data_path):
    os.mkdir(train_data_path)

hr_path = os.path.join(raw_data_path, 'hr/')
hr_deg_path = os.path.join(raw_data_path, 'hr_deg/')
lr_path = os.path.join(raw_data_path, 'lr/')

hr = []
hr_deg = []
lr = []

stride = 64
patch_size = 128

# signal_intensity_threshold=600  #parameter for selecting image patches containing signals
signal_intensity_threshold=0  #parameter for selecting image patches containing signals

hr_interval=1
lr_interval=3

for i in range(0, len(os.listdir(hr_path)), hr_interval):

    hr_img = tifffile.imread(hr_path + str(i) + '.tif')
    hr_deg_img = tifffile.imread(hr_deg_path + str(i) + '.tif')
    print(i + 1)

    # for m in range(0, hr_img.shape[0] - patch_size + 1, stride):
    #     for n in range(0, hr_img.shape[1] - patch_size + 1, stride):
    #         crop_hr = hr_img[m:m + patch_size, n:n + patch_size]
    #         crop_hr_deg = hr_deg_img[m:m + patch_size, n:n + patch_size]

    #         if np.max(crop_hr >= signal_intensity_threshold):
    #             hr.append(crop_hr)
    #             hr_deg.append(crop_hr_deg)

    hr.append(hr_img)
    hr_deg.append(hr_deg_img)

for i in range(0, len(os.listdir(lr_path)), lr_interval):
    lr_img = tifffile.imread(lr_path + str(i) + '.tif')
    print(i + 1)

    # for m in range(0, lr_img.shape[0] - patch_size + 1, stride):
    #     for n in range(0, lr_img.shape[1] - patch_size + 1, stride):
    #         crop_lr = lr_img[m:m + patch_size, n:n + patch_size]

    #         if np.max(crop_lr >= signal_intensity_threshold):
    #             lr.append(crop_lr)

    lr.append(lr_img)

hr = np.asarray(hr, dtype=np.float32)
hr_deg = np.asarray(hr_deg, dtype=np.float32)
lr = np.asarray(lr, dtype=np.float32)
print(hr.shape, hr_deg.shape, lr.shape)

np.savez(path + '/train_data/train_data.npz', hr=hr, hr_deg=hr_deg, lr=lr)