import torch
import random
from torch.utils.data import Dataset
import numpy as np


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 3:
        return np.rot90(img, k=2)
    elif mode == 4:
        return np.flipud(np.rot90(img))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def transform_input_img(img,mode,min_v,max_v):
    img = augment_img(img,mode)
    img=np.array(img,dtype=np.float32)
    img=(img-min_v)/(max_v-min_v)
    img[img>1]=1
    img[img<0]=0
    img=np.expand_dims(img,axis=0)
    return torch.from_numpy(img)

def transform_target_img(img,mode,min_v,max_v):
    img=augment_img(img,mode)
    img=np.array(img,dtype=np.float32)
    img = (img-min_v)/ (max_v-min_v)
    img[img>1]=1
    img[img < 0] = 0
    img=np.expand_dims(img,axis=0)
    return torch.from_numpy(img)


class ImageDataset_numpy(Dataset):
    def __init__(self,dataset,paired,min_v,max_v):
        self.xy=dataset['xy']
        self.xy_lr=dataset['xy_lr']
        self.xz=dataset['xz']
        self.paired=paired
        self.min_v=min_v
        self.max_v=max_v

    def __getitem__(self, index):

        mode1=np.random.randint(0, 8)
        mode2=np.random.randint(0, 8)
        if self.paired:
            item_xz = transform_input_img(self.xz[(index)%(self.xz.shape[0])], mode1,self.min_v,self.max_v)
            item_xy = transform_target_img(self.xy[(index)%(self.xy.shape[0])], mode1,self.min_v,self.max_v)
            item_xy_lr = transform_target_img(self.xy_lr[(index) % (self.xy_lr.shape[0])], mode1,self.min_v,self.max_v)
        else:
            item_xz = transform_input_img(self.xz[random.randint(0, (self.xz).shape[0]- 1)], mode1,self.min_v,self.max_v)
            item_xy = transform_target_img(self.xy[(index)%(self.xy.shape[0])], mode2,self.min_v,self.max_v)
            item_xy_lr=transform_target_img(self.xy_lr[(index)%(self.xy_lr.shape[0])], mode2,self.min_v,self.max_v)

        return {'xz': item_xz, 'xy': item_xy, 'xy_lr':item_xy_lr}

    def __len__(self):
        return max((self.xz).shape[0], (self.xy).shape[0])


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def create_train_data(train_data_path,batch_size,min_v,max_v):

    dataset=np.load(train_data_path+'train_data.npz')
    train_data=torch.utils.data.DataLoader(ImageDataset_numpy(dataset,False,min_v,max_v),batch_size=batch_size,shuffle=True)

    print('done data preprocessing!')
    return train_data


