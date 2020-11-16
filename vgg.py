import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

vgg_type  = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

class VGGNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(VGGNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(vgg_type["vgg16"]) 
        self.fc_layers = self.create_fc_layers(vgg_type["vgg16"])
    
    def forward(self, x):
        x = self.conv_layers(x)
        lastly_conv_img_size = self.img_size(vgg_type["vgg16"])
        x = x.view(-1, 512*lastly_conv_img_size*lastly_conv_img_size)
        x = self.fc_layers(x)
        return x
    
    def img_size(self, architecture):
        num_maxpool = len(['M' for i in architecture if i=='M'])
        lastly_conv_img_size = int(224/(2**num_maxpool))
        return lastly_conv_img_size
    
    def create_fc_layers(self, architecture):
        lastly_conv_img_size = self.img_size(architecture)
        layers = [
            nn.Linear(in_features=512*lastly_conv_img_size*lastly_conv_img_size, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes),
            nn.ReLU()
        ]
        return nn.Sequential(*layers)

    def create_conv_layers(self, architecture):
        layers = []
        input_channels = self.input_channels
        for x in architecture:
            if type(x)==int:
                output_channels = x
                layers+=[nn.Conv2d(in_channels=input_channels, out_channels=output_channels, 
                                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                        nn.BatchNorm2d(x),
                        nn.ReLU()]
                input_channels = x
            else:
                layers+=[nn.MaxPool2d(kernel_size=(2, 2))] 
        return nn.Sequential(*layers)


                






