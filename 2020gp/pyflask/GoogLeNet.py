#import coutils
#from coutils import fix_random_seed

#from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score

#import matplotlib.pyplot as plt
#import numpy as np 
#import os
#import pandas as pd 
#import pickle

class Inception(nn.Module):
  
  def __init__(self,in_ch,out_ch11,mid_ch13,out_ch13,mid_ch133,out_ch133,out_pool):
      super(Inception,self).__init__()
        
      self.conv11 = nn.Sequential(
          nn.Conv2d(in_ch,out_ch11,1,1),
          nn.BatchNorm2d(out_ch11),
          nn.ReLU())
        
      self.conv13 = nn.Sequential(
          nn.Conv2d(in_ch,mid_ch13,kernel_size=1,stride=1),
          nn.BatchNorm2d(mid_ch13),
          nn.ReLU(),
          nn.Conv2d(mid_ch13,out_ch13,kernel_size=3,stride=1,padding=1),
          nn.BatchNorm2d(out_ch13),
          nn.ReLU())
    
      self.conv133 = nn.Sequential(
          nn.Conv2d(in_ch,mid_ch133,kernel_size=1,stride=1),
          nn.BatchNorm2d(mid_ch133),
          nn.ReLU(),
          nn.Conv2d(mid_ch133, out_ch133, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_ch133),
          nn.ReLU(),
          nn.Conv2d(out_ch133,out_ch133, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_ch133),
          nn.ReLU(),
        )
        
      self.pool_conv1 = nn.Sequential(
          nn.MaxPool2d(3,stride=1,padding=1),
          nn.Conv2d(in_ch,out_pool,kernel_size=1,stride=1),
          nn.BatchNorm2d(out_pool),
          nn.ReLU())
      
  def forward(self,x):
      conv11_out = self.conv11(x)
      conv13_out = self.conv13(x)
      conv133_out = self.conv133(x)
      pool_conv_out = self.pool_conv1(x)
      outputs = torch.cat([conv11_out,conv13_out,conv133_out,pool_conv_out],1) 
      return outputs

class GoogLeNet(nn.Module):  
  def __init__(self,num_classes):
    super(GoogLeNet,self).__init__()
    self.layer_1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7,stride=2, padding=1),
        nn.MaxPool2d(3,stride=2,padding=1),
        nn.Conv2d(64, 192, kernel_size=3, stride=1,padding=1),
        nn.MaxPool2d(3,stride=2,padding=1)
    )

    self.layer_2 = nn.Sequential(
        Inception(192,64,96,128,16,32,32),
        Inception(256,128,128,192,32,96,64),
        nn.MaxPool2d(3,stride=2,padding=1)
    )

    self.layer_3 = nn.Sequential(
        Inception(480,192,96,208,16,48,64),
        Inception(512,160,112,224,24,64,64),
        Inception(512,128,128,256,24,64,64),
        Inception(512,112,144,288,32,64,64),
        Inception(528,256,160,320,32,128,128),
        nn.MaxPool2d(3,stride=2,padding=1)
    )

    self.layer_4 = nn.Sequential(
        Inception(832,256,160,320,32,128,128),
        Inception(832,384,192,384,48,128,128),
        nn.AvgPool2d(7, stride=1)
    )

    self.dropout = nn.Dropout2d(0.4)
    self.fc_layer = nn.Linear(1024,num_classes)

  def forward(self,x):
    out = self.layer_1(x)
    #print(out.shape)
    out = self.layer_2(out)
    #print(out.shape)
    out = self.layer_3(out)
    #print(out.shape)
    out = self.layer_4(out)
    #print(out.shape)
    out = self.dropout(out)
    #print(out.shape)
    out = out.view(out.size(0),-1)
    #print(out.shape)
    out = self.fc_layer(out)
    
    return out