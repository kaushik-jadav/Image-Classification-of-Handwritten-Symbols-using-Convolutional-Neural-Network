#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_func(Dataset):
    
    
#     data=np.load('Dataset')
    labels=Dataset['train_labels'].T
    images=Dataset['train_images']

    
    def load_data(images,labels):
        X=images
        y=labels
        X=X.reshape(X.shape[0],1,150,150)
        X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.33, random_state=42)
        X_train=X_train/255
        X_test=X_test/255
        X_trainTensor=torch.Tensor(X_train)
        X_testTensor=torch.Tensor(X_test)
        y_trainTensor=torch.Tensor(y_train)
        y_testTensor=torch.Tensor(y_test)
        y_trainTensor=y_trainTensor.type(torch.LongTensor)
        y_testTensor=y_testTensor.type(torch.LongTensor)
        return X_trainTensor,y_trainTensor,X_testTensor,y_testTensor

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

    def train(X_train,y_train,X_test,y_test):
    #X_train,y_train,X_test,y_test=load_data("Train_Images.npy","Train_Labels.npy")
        print("Training Image Size: ",X_train.shape)
        print("Training Labels Size: ",y_train.shape)
        print("Test Image Size: ",X_test.shape)
        print("Test Labels Size: ",y_test.shape)
        train_loader=DataLoader(TensorDataset(X_train,y_train),batch_size=10,shuffle=True) # create your dataloader
        test_loader=DataLoader(TensorDataset(X_test,y_test),batch_size=10,shuffle=True)

        net=ConvolutionalNetwork()
        criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(net.parameters(), lr=0.001)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


        start_timer_=time.time()
        no_of_epochs= 150
        max_training_batch=3000
        max_testing_batch=3000
        loss_train_data=[]
        loss_test_data=[]
        train_no_correct=[]
        test_no_correct=[]

        for ii in range(no_of_epochs):
            train_corr=0
            test_corr=0

            # Run the training batches
            for bb, (X_images, Y_labels) in enumerate(train_loader):

                # Limit the number of batches
                if bb == max_training_batch:
                    break
                bb+=1

                # Apply the model
                y_pred=net(X_images)
                loss=criterion(y_pred, Y_labels)

                # Tally the number of correct predictions
                predicted_data=torch.max(y_pred.data, 1)[1]
                batch_corr=(predicted_data == Y_labels).sum()
                train_corr += batch_corr

                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print interim results
                if bb%100 == 0:
                    print(f'epoch: {ii:2}  batch: {bb:4} [{10*bb:6}/15000]  loss: {loss.item():10.8f} accuracy: {train_corr.item()*100/(10*bb):7.3f}%')

            loss_train_data.append(loss)
            train_no_correct.append(train_corr)

            # Run the testing batches
            true_values=[]
            predictions=[]
            with torch.no_grad():
                for bb, (X_images, Y_labels) in enumerate(test_loader):
                    # Limit the number of batches
                    if bb == max_testing_batch:
                        break

                    # Apply the model
                    y_val=net(X_images)

                    # Tally the number of correct predictions
                    predicted_data=torch.max(y_val.data, 1)[1] 
                    test_corr += (predicted_data == Y_labels).sum()
                    predictions.append(predicted_data.tolist())
                    true_values.append(Y_labels.tolist())

            loss=criterion(y_val, Y_labels)
            loss_test_data.append(loss)
            test_no_correct.append(test_corr)

        print(f'\nDuration: {time.time() - start_timer_:.0f} seconds') # print the time elapsed
        print("Saving model...")
        torch.save(net.state_dict(), 'model.ckpt')
        print("Model saved as 'model.ckpt'")

        plt.figure(figsize=(25,25))
        plt.subplot(211)
        plt.plot([t/157.46 for t in train_no_correct], label='training accuracy')
        plt.plot([t/77.56 for t in test_no_correct], label='validation accuracy')
        plt.title('Accuracy at the end of each epoch')
        plt.legend()

        train_loss=[]
        test_loss=[]
        for x in loss_train_data:
            train_loss.append(x.item())
        for y in loss_test_data:
            test_loss.append(y.item())
        plt.subplot(212)
        plt.plot(train_loss, label='training loss')
        plt.plot(test_loss, label='validation loss')
        plt.title('Loss at the end of each epoch')
        plt.legend()

        plt.show()
    

    
    def evaluate(X,y):
        model=ConvolutionalNetwork()
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        predictions=[]
        test_loader=DataLoader(TensorDataset(X,y),batch_size=10,shuffle=True)
        with torch.no_grad():
            correct=0
            total=0
            for images, labels in test_loader:
                images=images.to(device)
                labels=labels.to(device)
                outputs=model(images)
        #         print(outputs.shape)
        #         for image in outputs:
        #             predictions.append(torch.argmax(image).item())
                _, predicted_data=torch.max(outputs.data, 1)
                predictions.append(predicted_data.tolist())
                total += labels.size(0)
                correct += (predicted_data == labels).sum().item()
                accuracy=(100 * correct / total)

            print('Test Accuracy of the model: {} %'.format(accuracy))

        if accuracy>95:
            # Save 
            torch.save(net.state_dict(), 'model.ckpt')
    

    def Val_func(X):
        predicted_labels=[]
        X=X.reshape(X.shape[0],1,150,150)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # create your dataloader
        dataloader=DataLoader(torch.Tensor(X),batch_size=10, shuffle=False)
        model=ConvolutionalNetwork()
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()
        with torch.no_grad():
            for images in dataloader:
                #print(images)
                images=images.to(device)
                outputs=model(images)
                _, predicted_data=torch.max(outputs.data, 1)
                #print(predicted_data)
                predicted_labels.append(predicted_data.tolist())
        #print(predicted_labels)
        predicted_labels=[item for sublist in predicted_labels for item in sublist]
            #     true_values=[item for sublist in true_values for item in sublist]    
        return predicted_labels

    
    def plot_confusionMatrix(predictions, true_values):
        conf_matrix=confusion_matrix(np.array(predictions), np.array(true_values))
        plt.figure(figsize=(25,25))
        sns.set(font_scale=1.8)
        sns.heatmap(conf_matrix, annot=True, fmt='g')

    
    
    X_train,y_train,X_test,y_test=load_data(images, labels)
    train(X_train,y_train,X_test,y_test)
    evaluate(X_test,y_test)
    predicted_val=Val_func(X_test)
    plot_confusionMatrix(predicted_val, y_test)

    
    return predicted_val

    
    
    
Dataset=np.load('training_dataset_zip.npz')
Y=train_func(Dataset)
    
    

