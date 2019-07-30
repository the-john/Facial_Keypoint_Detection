## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        # Looking to stay as close as possible to the NaimishNet model in the published paper.  Image size is [1, 96, 96]
        self.conv1 = nn.Conv2d(1, 32, 4)       # goes to (32, 93, 93); w/ maxpool goes to (32, 46, 46), round down
        self.conv2 = nn.Conv2d(32, 64, 3)      # goes to (64, 44, 44); w/ maxpool goes to (64, 22, 22)
        self.conv3 = nn.Conv2d(64, 128, 2)     # goes to (128, 22, 22); w/ maxpool goes to (128, 11, 11)
        self.conv4 = nn.Conv2d(128, 256, 1)    # goes to (256, 11, 11); w/ maxpool goes to (256, 5, 5), round down
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        #self.fc1 = nn.Linear((256 * 14 * 14), 1000) 
        self.fc1 = nn.Linear((256 * 5 * 5), 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers 
        # (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        #flatten
        #x = x.view(-1, (256 * 14 * 14))
        x = x.view(-1, (256 * 5 * 5))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
