import torch
import torch.nn as nn   
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,i_downsample =None, stride=1):
        super(Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)  
        self.i_downsample = i_downsample    
        self.stride = stride
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x.clone()    
        x   = self.relu(self.batchnorm2(self.conv1(x)))
        x   = self.batchnorm2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self,Block:Block,layer_list,num_classes,num_channels=3):
        super(ResNet,self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(Block,layer_list[0],out_channels=64,stride=1)
        self.layer2 = self._make_layer(Block,layer_list[1],out_channels=128,stride=2)
        self.layer3 = self._make_layer(Block,layer_list[2],out_channels=256,stride=2)
        self.layer4 = self._make_layer(Block,layer_list[3],out_channels=512,stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*Block.expansion,num_classes)
        
    def forward(self,x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
    def _make_layer(self,Block,num_residual_blocks,out_channels,stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels*Block.expansion:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels*Block.expansion,kernel_size=1,stride=stride),
                                                nn.BatchNorm2d(out_channels*Block.expansion))
        layers.append(Block(self.in_channels,out_channels,identity_downsample,stride))
        self.in_channels = out_channels * Block.expansion
        
        for i in range(num_residual_blocks-1):
            layers.append(Block(self.in_channels,out_channels))
        return nn.Sequential(*layers)
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,i_downsample=None,stride=1):
        super(BottleNeck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0,bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
def ResNet50(num_classes,img_channel=3):
    return ResNet(BottleNeck,[3,4,6,3],num_classes,num_channels=img_channel)
def ResNet101(num_classes,img_channel=3):
    return ResNet(BottleNeck,[3,4,23,3],num_classes,num_channels=img_channel)
def ResNet152(num_classes,img_channel=3):
    return ResNet(BottleNeck,[3,8,36,3],num_classes,num_channels=img_channel)
    