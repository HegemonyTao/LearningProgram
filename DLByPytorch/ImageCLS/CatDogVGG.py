from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import Dataset,DataLoader
import time
#使用VGG对CatDogCLSManula.py中的模型进行迁移学习
#设置CUDA
is_cuda=False
if torch.cuda.is_available():
    is_cuda=True
#加载数据并格式化
simple_transform = transforms.Compose([transforms.Resize((224,224))
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])
#对训练数据采用数据增强
train_transform=transforms.Compose([transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip()
                                    ,transforms.RandomRotation(0.2),
                                    transforms.ToTensor()
                                    ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('../data/train/',train_transform)
valid = ImageFolder('../data/valid/',simple_transform)
#加载数据到DataLoader
train_data_loader = torch.utils.data.DataLoader(train,batch_size=32,num_workers=3,shuffle=True)
valid_data_loader = torch.utils.data.DataLoader(valid,batch_size=32,num_workers=3,shuffle=True)
#加载VGG参数
vgg=models.vgg16(pretrained=True)
vgg=vgg.cuda()
vgg.classifier[6].out_features=2#将输出类别设置为2
for param in vgg.features.parameters():#冻结梯度
    param.requires_grad=False
#定义优化器进行学习
optimizer = optim.SGD(vgg.classifier.parameters(),lr=0.0001,momentum=0.5)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 500)
        self.fc2 = nn.Linear(500,50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
#定义训练函数
def fit(epoch,model,data_loader,phase='training',volatile=False):
    #在训练和测试时，对模型采用不同的策略
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        #训练阶段梯度设为0
        if phase == 'training':
            optimizer.zero_grad()
        #模型输出+计算损失
        output = model(data)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, size_average=False).item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        #反向求导+梯度下降
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy
model=Net()
if is_cuda:
    model.cuda()
if __name__=='__main__':
    #调整dropout层的丢失比率
    for layer in vgg.classifier.children():
        if type(layer)==nn.Dropout():
            layer.p=0.2
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 10):
        epoch_loss, epoch_accuracy = fit(epoch, vgg, train_data_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, vgg, valid_data_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)