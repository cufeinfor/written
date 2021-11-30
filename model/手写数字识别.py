
import torch
import numpy as np
import os
from torch import nn
from torch import optim
from torch._C import TracingState
#引入激活函数
import torch.nn.functional as F
from torch.optim import optimizer

from torchvision import datasets
from torchvision.datasets import MNIST, mnist
from torch.utils.data import DataLoader
from torchvision.transforms import Compose,ToTensor,Normalize


#准备数据
BATCH_SIZE=128
test_batch_size=1000

def get_dataloader(train=True,batch_size=test_batch_size):
    transform_fn=Compose([ToTensor(),Normalize(mean=(0.1307,),std=(0.3081,))])
    dataset=MNIST(root="C:/Users/qiaoy/Desktop/data",train=train,transform=transform_fn)
    data_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    return data_loader

#构建模型
class MnistNet(nn.Module):
    def __init__(self) -> None:
        super(MnistNet,self).__init__()
        #定义Linear输入和输出的形状
        self.fc1=nn.Linear(28*28*1,28)
        self.fc2=nn.Linear(28,10)
    def forward(self,input):
        #对数据形状变形
        input =input.view(-1,28*28*1)
        input=self.fc1(input)
        input=F.relu(input)
        out=self.fc2(input)
        #取交叉熵损失
        return F.log_softmax(out)

#实例化模型
mnist_net=MnistNet()
optimizer=optim.Adam(mnist_net.parameters(),lr=0.001)

if os.path.exists("./model/model.pkl"):
  mnist_net.load_state_dict(torch.load("C:/Users/qiaoy/Desktop/written/model/model.pkl"))
  optimizer.load_state_dict(torch.load("C:/Users/qiaoy/Desktop/written/model/optimizer.pkl"))
#实现训练过程
def train(epoch): 
    mode=True #模型设置为训练模型
    mnist_net.train(mode=mode)
    data_loader=get_dataloader(train=mode)#获取训练数据集
    #训练每一轮的参数
    for idx,(input,traget) in enumerate(data_loader):
        optimizer.zero_grad()#梯度获取为0
        output=mnist_net(input)#进行向前计算
        loss=F.nll_loss(output,traget)#带权损失
        loss.backward()#反向传播
        optimizer.step()#参数更新
        if idx%100==0:
            print(epoch,idx,loss.item())

        #模型的保存
        if idx%100==0:
            torch.save(mnist_net.state_dict(),"C:/Users/qiaoy/Desktop/written/model/model.pkl")#保存模型参数
            torch.save(optimizer.state_dict(),"C:/Users/qiaoy/Desktop/written/model/optimizer.pkl")#保存优化器


def test():
    loss_list=[]
    acc_list=[]
    mnist_net.eval()#设置为评估模式
    test_dataloader=get_dataloader(train=False)#获取评估数据集
    for idx,(input ,target) in enumerate(test_dataloader):
        with torch.no_grad():#不计算梯度
            output =mnist_net(input)#形状是[batch_size,10],target形状是[batch_size]
            cur_loss=F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算准确率
            pred=output.max(dim=-1)[-1]#最大值的位置,获取预测值
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))


if __name__=='__main__':
    test()

    #训练3轮
    #for i in range(3):
    #     train(i)

    """loader=get_dataloader(train=False)
    for input, lable in loader:
        print(lable.size())
        break """
