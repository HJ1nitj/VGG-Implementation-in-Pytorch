import os
import pandas as pd
import numpy as np
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dataloader import LoadImages
from vgg import VGGNet

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print (device)

# reading json
content = open("config.json", 'r').read()
content = json.loads(content)
TRAIN_IMAGE_DIR = content['TRAIN_IMAGE_DIR']
TEST_IMAGE_DIR = content['TEST_IMAGE_DIR']
TRAIN_LABEL_DIR = content['TRAINING_LABEL_FILE_PATH']
TEST_LABEL_DIR = content['TRAINING_LABEL_FILE_PATH']
N_EPOCHS = content['N_EPOCHS']


# Dataloader: TRAIN
train_data = LoadImages(img_dir=TRAIN_IMAGE_DIR, label_dir=TRAIN_LABEL_DIR, transform=transforms.ToTensor())
trainloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=0)
num_classes = 16#len(train_data)
print (num_classes)
# Dataloader: TEST
test_data = LoadImages(img_dir=TEST_IMAGE_DIR, label_dir=TEST_LABEL_DIR, transform=transforms.ToTensor())
testloader = DataLoader(dataset=test_data, batch_size=16, shuffle=True, num_workers=0)

# VGG
vgg_net = VGGNet(1, num_classes=num_classes)
vgg_net = vgg_net.float()#.to(device)

# Loss
criterion = RMSELoss()
optimizer = optim.Adam(params=vgg_net.parameters(), lr=0.0001)

# Number of iterations
total_samples = len(train_data)
n_iterations = math.ceil(total_samples/16)

print (total_samples)

# Training
mean_losses = []
for epoch in range(N_EPOCHS):
    count=1
    running_loss = []
    print ('Epoch {}'.format(epoch))
    for i, data in enumerate(trainloader):
        images, labels = data# data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = vgg_net(images.float())
        #loss
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()
        if count<=n_iterations:
            print ("\t", "{}/{} --- Loss={}".format(count, n_iterations, loss.item()))
            running_loss.append(loss.item())
            count=count+1
    mean_losses.append(np.mean(running_loss))

print (mean_losses)
print ('Mean Loss: {}'.format(np.mean(mean_losses)))

PATH = r"C:\Users\hj21904\Documents\Deep learning practice\pytroch\pretrained/vgg16_net.pth"
torch.save(vgg_net.state_dict(), PATH)
