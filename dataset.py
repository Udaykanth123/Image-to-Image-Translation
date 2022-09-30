import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import skimage.io
import numpy as np
import torchvision.transforms as transforms
from skimage.transform import resize


def normalize():
    return  transforms.Compose([
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



class load_data(Dataset):
    def __init__(self,img_path1,img_path2):
        super(load_data,self).__init__()
        images_path1=[g for g in os.listdir(img_path1) if g.endswith(".png")]
        images_path2=[g for g in os.listdir(img_path2) if g.endswith(".jpg")]
        self.all_img_paths1=[]
        self.all_img_paths2=[]
        for i in range(len(images_path1)):
            self.all_img_paths1.append(os.path.join(img_path1,images_path1[i]))
        for i in range(len(images_path2)):
            self.all_img_paths2.append(os.path.join(img_path2,images_path2[i]))
        self.normalise=normalize()
    
    def __len__(self):
        return min(len(self.all_img_paths1),len(self.all_img_paths2))
    
    def __getitem__(self,id):
        img1=skimage.io.imread(self.all_img_paths1[id])/255.0
        img2=skimage.io.imread(self.all_img_paths2[id])/255.0
        img1=resize(img1,(64,64))
        img2=resize(img2,(64,64))
        # print(np.max(img))
        img1=img1.transpose(2, 0, 1)
        img2=img2.transpose(2, 0, 1)
        # img=np.expand_dims(img,axis=0)
        img1=torch.from_numpy(img1.astype(np.float32))
        img2=torch.from_numpy(img2.astype(np.float32))
        img1=self.normalise(img1)
        img2=self.normalise(img2)
        # print(torch.min(img))
        return img1,img2


        
        
