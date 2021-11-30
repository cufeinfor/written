#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 定义模型
@Date     :2021/11/20 14:35:58
@Author      :Amanda
@version      :1.0
'''

from pickle import encode_long, load
from token import ENCODING
import torch.nn as nn
# from libs import *
# from libs import Max_len
# from libs import wss
import libs
import torch.nn.functional as F
from torch.nn.modules import loss, module
from torch.optim import Adam
# from dataset import Dataset,DataLoader
from dataset import *
from libs import Max_len
import numpy as np


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.embedding=nn.Embedding(len(libs.model_loader),200)#[词的数量,the size of each embedding vector]
        self.fc=nn.Linear(200,2)#[max_len,类别的数量]#全连接层,二分类


    def forward(self,input):
        """
        @description  :
        ---------
        @param  :input的形状是[batch_size,max_len]
        -------
        @Returns  :
        -------
        """
        #进行embedding操作，
        x=self.embedding(input)#embeding后，x是[batch_size,max_len,the size of each embedding vector]，3维的
        x.view([-1,libs.Max_len*100])#换成2维的
        out=self.fc(x)
        return F.log_softmax(out,dim=-1)
    

model=MyModel()
optimazer=Adam(model.parameters(),0.001)

def train(epoch):
    for idx,(input,target) in enumerate(get_dataloader(train=True)):
        #梯度归0
        optimazer.zero_grad()
        output=model(input)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimazer.step()

        print(loss.item())
        print(idx,input,target)

if __name__=='__main__': 
    for i in range(1):
        train(i)
    # for idx,(input,target) in enumerate(get_dataloader(train=True)):
    #    print("input:",input)
    #    print("inputsize:",Tensor.size(input))
    #    output= model(input)
    #    print("OUTputsize:",output)
    #    print("OUTputsize:",Tensor.size(output))
    #     output=model(input)
    #     print(output)
    #     print(torch.Size(output))
    #     break

