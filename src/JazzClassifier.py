#JazzClassifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class JazzClassifierCNN(nn.Module): # This is the pre-trained style critic
    def __init__(self, in_channels=4): 
        super(JazzClassifierCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels, 32, kernel_size=(3,5), padding=(1,2))
        # self.conv1=nn.Conv2d(4,32,kernel_size=(3,5),padding=(1,2))
        self.bn1=nn.BatchNorm2d(32)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(64);self.pool2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=2,dilation=2)
        self.bn3=nn.BatchNorm2d(128)
        self.global_avg_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(128,256);self.dropout1=nn.Dropout(0.5)
        self.fc2=nn.Linear(256,128);self.dropout2=nn.Dropout(0.5)
        self.fc_out=nn.Linear(128,1)
    def forward(self,x):
        x=self.pool1(F.relu(self.bn1(self.conv1(x))))
        x=self.pool2(F.relu(self.bn2(self.conv2(x))))
        x=F.relu(self.bn3(self.conv3(x)))
        x=self.global_avg_pool(x)
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=self.dropout1(x)
        x=F.relu(self.fc2(x))
        x=self.dropout2(x)
        return self.fc_out(x)
    

class JazzClassifierCNN_3Channel(nn.Module):
    """
    This is the 3-CHANNEL version of the classifier.
    It must match the architecture of the model you trained.
    """
    def __init__(self, num_classes=1):
        # ***** FIX: Corrected the super() call to the right class name *****
        super(JazzClassifierCNN_3Channel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 5), stride=1, padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        return self.fc_out(x)
