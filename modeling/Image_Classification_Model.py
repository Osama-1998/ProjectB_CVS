# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD



class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        #fist layer
        layer1 =  Conv2d(16, 8, kernel_size=5, stride=1, padding=1)
        layer1.append( BatchNorm2d(4))
        layer1.append(ReLU(inplace=True))
        layer1.append(MaxPool2d(kernel_size=2, stride=2))

        #second layer
        layer2 =  Conv2d(8, 8, kernel_size=5, stride=1, padding=1)
        layer2.append( BatchNorm2d(4))
        layer2.append(ReLU(inplace=True))
        layer2.append(MaxPool2d(kernel_size=2, stride=2))

        # third layer
        layer3 = Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        layer3.append(BatchNorm2d(4))
        layer3.append(ReLU(inplace=True))
        layer3.append(MaxPool2d(kernel_size=2, stride=2))

        # fourth layer
        layer4 = Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        layer4.append(BatchNorm2d(4))
        layer4.append(ReLU(inplace=True))
        layer4.append(MaxPool2d(kernel_size=2, stride=2))

        # fifth layer
        layer5 = Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        layer5.append(BatchNorm2d(4))
        layer5.append(ReLU(inplace=True))
        layer5.append(MaxPool2d(kernel_size=2, stride=2))

        ((W - K + 2P) / S) + 1)
        Here
        W = Input
        size
        K = Filter
        size
        S = Stride
        P = Padding
        linear_layer = Linear(4 * 7 * 7, 10);


        self.cnn_layers = Sequential(
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x