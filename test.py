#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from google_drive_downloader import GoogleDriveDownloader as gdd

# https://drive.google.com/file/d/1vQLLGjVa2HAcxUba2o-C1SLpMAlPyC9u/view?usp=sharing

def test_func(X):
    gdd.download_file_from_google_drive( file_id='1vQLLGjVa2HAcxUba2o-C1SLpMAlPyC9u',
                                    dest_path='../trained-model.ckpt',
                                    unzip=False, showsize = True)
    predicted_labels=[]
    X=X.reshape(X.shape[0],1,150,150)
    X=X/255
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class ConvolutionalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=nn.Conv2d(1, 6, (3,3), 1)
            self.conv2=nn.Conv2d(6, 16, (3,3), 1)
            self.conv3=nn.Conv2d(16, 32, (3,3), 1)
            self.conv4=nn.Conv2d(32, 96, (3,3), 1)
    #         self.conv1=nn.Conv2d(1, 6, (3,3), 1)
    #         self.conv2=nn.Conv2d(6, 16, (3,3), 1)
    #         self.conv3=nn.Conv2d(16, 30, (3,3), 1)
    #         self.conv4=nn.Conv2d(30, 48, (3,3), 1)
            self.fc1=nn.Linear(7*7*96, 84)
            self.fc2=nn.Linear(84, 84)
            self.fc3=nn.Linear(84,84)
            self.fc4=nn.Linear(84,25)

        def forward(self, X):
            X=F.relu(self.conv1(X))
            X=F.max_pool2d(X, 2, 2)
            X=F.relu(self.conv2(X))
            X=F.max_pool2d(X, 2, 2)
            X=F.relu(self.conv3(X))
            X=F.max_pool2d(X, 2, 2)
            X=F.relu(self.conv4(X))
            X=F.max_pool2d(X, 2, 2)
            X=X.view(-1, 7*7*96)
            X=F.relu(self.fc1(X))
            X=F.relu(self.fc2(X))
            X=F.relu(self.fc3(X))
            X=self.fc4(X)
            return F.log_softmax(X, dim=1)
    # create your dataloader
    dataloader=DataLoader(torch.Tensor(X),batch_size=10, shuffle=False)
    model=ConvolutionalNetwork()
    model.load_state_dict(torch.load('../trained-model.ckpt'))
    model.eval()
    with torch.no_grad():
        for images in dataloader:
            #print(images)
            images=images.to(device)
            outputs=model(images)
            _, predicted=torch.max(outputs.data, 1)
            #print(predicted)
            predicted_labels.append(predicted.tolist())
    #print(predicted_labels)
    predicted_labels=[item for sublist in predicted_labels for item in sublist]
        #     true_values=[item for sublist in true_values for item in sublist]    
    return predicted_labels


# Now load the labels here for checking the accuracy of the test data.
test_images=np.load('Path for Test Images')
predicted_val=test_func(test_images)
labels=np.load('Path for Test Labels').T
true_val=labels
correct_count=0
for i in range(len(true_val)):      
      if(predicted_val[i]==true_val[i]):
             correct_count += 1   
accuracy=(correct_count / len(true_val))*100 
print('Accuracy on Test Data = ', accuracy)


# In[ ]:




