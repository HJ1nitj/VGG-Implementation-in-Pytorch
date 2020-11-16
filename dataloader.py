import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class LoadImages(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224, transform=None, training=True, testing=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size
        self.training = training
        self.testing = testing
        # listing all the image names present in a directory
        self.img_names = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
        # loading training file where target values are present
        if training:
            self.label_df = pd.read_csv(label_dir+'/training.csv')
        elif testing:
            self.label_df = pd.read_csv(label_dir+'/testing.csv')
        
        self.lh_column_name = [col for col in self.label_df.columns if col.split('_')[0]=='LH' and 'E' in col.split('_')]
        self.lh_score = self.label_df[self.lh_column_name].values
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_label = self.lh_score[index]
        # opening image
        img = np.asarray(ImageOps.grayscale(Image.open(img_name).resize((self.img_size, self.img_size))))/255

        if self.transform:
            img = self.transform(img)
        return img, img_label
    
    def __len__(self):
        return len(self.img_names)
    

        



