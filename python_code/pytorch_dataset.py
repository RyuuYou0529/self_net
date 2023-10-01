import torch
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

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

def transform_input_img(img,mode):
    img = augment_img(img,mode)
    # img = np.asarray(img, dtype=np.float32)
    img=np.expand_dims(img,axis=0)
    return torch.from_numpy(img.copy())

def transform_target_img(img,mode):
    img=augment_img(img,mode)
    # img = np.asarray(img, dtype=np.float32)
    img=np.expand_dims(img,axis=0)
    return torch.from_numpy(img.copy())

class ImageDataset_numpy(Dataset):
    def __init__(self, dataset, paired, normalize_mode):
        if normalize_mode == 'min_max':
            self.normalize = lambda t:(t-t.min())/(t.max()-t.min())
        elif normalize_mode == 'z_score':
            self.normalize = lambda t:(t-t.mean())/t.std()
        else:
            self.normalize = lambda t:t
        
        self.hr=self.normalize(dataset['hr'])
        self.hr_deg=self.normalize(dataset['hr_deg'])
        self.lr=self.normalize(dataset['lr'])

        self.paired=paired

        self.vol_sum = np.sum(self.lr)

    def __getitem__(self, index):
        mode1=np.random.randint(0, 8)
        mode2=np.random.randint(0, 8)
        if self.paired:
            item_lr = transform_input_img(self.lr[(index)%(self.lr.shape[0])], mode1)
            item_hr = transform_target_img(self.hr[(index)%(self.hr.shape[0])], mode1)
            item_hr_deg = transform_target_img(self.hr_deg[(index) % (self.hr_deg.shape[0])], mode1)
        else:
            item_lr = transform_input_img(self.lr[np.random.randint(0, (self.lr).shape[0])], mode1)
            # item_lr = transform_input_img(self.lr[(index)%(self.lr.shape[0])], mode1)
            item_hr = transform_target_img(self.hr[(index)%(self.hr.shape[0])], mode2)
            item_hr_deg=transform_target_img(self.hr_deg[(index)%(self.hr_deg.shape[0])], mode2)

        return {'lr':item_lr, 'hr':item_hr, 'hr_deg':item_hr_deg}

    def __len__(self):
        return max((self.lr).shape[0], (self.hr).shape[0])


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

def create_train_data(train_data_path, batch_size, normalize_mode: str='min_max'):
    dataset=np.load(os.path.join(train_data_path, 'train_data.npz'))
    train_data=DataLoader(ImageDataset_numpy(dataset,False,normalize_mode),batch_size=batch_size,shuffle=True)

    print('done data preprocessing!')
    return train_data


