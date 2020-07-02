import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import pdb

#Extending PyTorch’s nn.Module Class
#Hyperparameters are parameters whose values are chosen manually and arbitrarily.
#kernel_size--The words kernel and filter are interchangeable.
#out_channels--One filter produces one output channel.
#out_features--Sets the size of the output tensor.
class Network(nn.Module):
  def __init__(self):
    super().__init__()
    #the in_channels of the first convolutional layer depend on the number of 
    #color channels present inside the images that make up the training set. 
    #Since we are dealing with grayscale images, we know that this value should be a 1.
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    #the input to one layer is the output from the previous layer, and so all of 
    #the in_channels in the conv layers and in_features in the linear layers 
    #depend on the data coming from the previous layer.
    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    #The out_features for the output layer depend on the number of classes that 
    #are present inside our training set. Since we have 10 classes of clothing 
    #inside the Fashion-MNIST dataset, we know that we need 10 output features.
    self.out = nn.Linear(in_features=60, out_features=10)
  def forward(self, t):
    # (1) input layer
    t = t

    # (2) hidden conv layer
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (3) hidden conv layer
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (4) hidden linear layer
    t = t.reshape(-1, 12 * 4 * 4)
    t = self.fc1(t)
    t = F.relu(t)

    # (5) hidden linear layer 
    t = self.fc2(t)
    t = F.relu(t)

    # (6) output layer
    t = self.out(t)
    #t = F.softmax(t, dim=1)

    return t

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
            transforms.ToTensor()
    ])
)

##Preparing For The Forward Pass
network = Network()#Creating an instance of our Network class.

train_loader = torch.utils.data.DataLoader(#Creating a data loader that provides batches of size 100 from our training set.
    train_set, 
    batch_size=100,
    shuffle=True
)
optimizer = optim.Adam(network.parameters(), lr=0.01)#Define the optim

tb = SummaryWriter()

for epoch in range(2):
  
  total_loss = 0
  total_correct = 0
  
  for batch in train_loader: # Get Batch
    images, labels = batch

    preds = network(images) # Pass Batch
    loss = F.cross_entropy(preds, labels) #  Calculating the loss

    optimizer.zero_grad()#zero out these gradients
    #We specifically need the gradient calculation feature anytime we are going 
    #to calculate gradients using the backward() function. Otherwise, it is a 
    #good idea to turn it off because having it off will reduce memory consumption 
    #for computations, e.g. when we are using networks for predicting (inference).
    #Enable PyTorch's gradient tracking feature
    #torch.set_grad_enabled(True)
    loss.backward() # Calculate Gradients
    optimizer.step() # Update Weights

    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)

    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
    
    tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    tb.add_histogram(
        'conv1.weight.grad'
        ,network.conv1.weight.grad
        ,epoch
    )

  print(
      "epoch:", epoch, 
      "total_correct:", total_correct, 
      "loss:", total_loss,
      "rate:", total_correct / len(train_set)
  )

tb.close()
