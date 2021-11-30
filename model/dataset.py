
import jieba
import torch
from torch._C import Size
from torch.functional import Tensor

from torch.utils.data import DataLoader,Dataset
import tokenize
import os
#正则表达式
import re
#去除英文标点
import string
from string import punctuation
#去掉中文标点
import zhon
from zhon.hanzi import punctuation
import libs
# from libs import *
# from libs import Max_len
# from libs import ws

#准备数据
data_base_path=r"C:/Users/qiaoy/Desktop/data/aclImdb"

#定义分词



def Mytokenize(content):
    chinese_punctuation=zhon.hanzi.punctuation
    english_punctuation=string.punctuation
    Allpunctuation=chinese_punctuation+english_punctuation
    # filters=["\\",':','!']
    # filters.append(Allpunctuation)
    #替换re.sub(pattern：模式字符串, repl：被替换成的字符串, string：要被替换的字符串, count=0, flags=0)
    temp_text=re.sub("[{}]".format(Allpunctuation),"",content,flags=re.S)#re.S,表示正则表达式会将这个字符串作为一个个整体，在整体中进行匹配
    tokens= [i.strip().lower() for i in temp_text.split()]
    return tokens



#读取文件
class Mydataset(Dataset):
    def __init__(self,train=True):
        self.train_path=r"C:/Users/qiaoy/Desktop/data/aclImdb/train"
        self.test_path=r"C:/Users/qiaoy/Desktop/data/aclImdb/test"
        data_path=self.train_path if train else self.test_path

        #把所有文件名放到列表里
        tem_data_path=[os.path.join(data_path,"pos"),os.path.join(data_path,"neg")]
        self.total_file_path=[]#所有评论文件的path
        for path in tem_data_path:
            file_name_list=os.listdir(path)
            file_path_list=[os.path.join(path,i) for i in file_name_list if i.endswith(".txt")]
            self.total_file_path.extend(file_path_list)



    def __getitem__(self, index):
        current_file_path=self.total_file_path[index]
        current_filename=os.path.basename(current_file_path)
        label_str=current_file_path.split("\\")[-2]
        label=0 if label_str=="neg" else 1
        #获取内容
        content =Mytokenize(open(current_file_path,encoding='utf-8').read())
        # content=torch.LongTensor(content)
        # label=torch.LongTensor(label)
        return content,label
        print(content.type())
        # label=int(current_filename.split("_"[-1].split(".")[0]))-1#处理标题，获取lable,转化为[0-9]
        # text=tokenize(open(current_path).read().strip())#直接按照空格进行分词
        # return label,text


    def __len__(self):
        return len(self.total_file_path)

def collate_fn(batch):
    content,label=list(zip(*batch))
    content=[libs.model_loader.transform(i,max_len=libs.Max_len) for i in content]
    content=torch.LongTensor(content)
    label=torch.LongTensor(label)
    return content,label


def get_dataloader(train=True):
#实例化，准备dataloader
   my_dataset=Mydataset(train)
#    dataloader=DataLoader(my_dataset)
   dataloader=DataLoader(my_dataset,batch_size=128,shuffle=True,collate_fn=collate_fn)
   return dataloader

if __name__=='__main__':
    for index,(label,target) in enumerate(get_dataloader()):
        # print("index:",index)
        # print("*-"*10)
        # print("label:",label)
        # print("labeltype:",type(label))
        # print("*-"*10)
        print("text:",target)
        print("targettype:",type(target))
        print("targetsize:",Tensor.size(target))
        
        break
    # get_dataloader(train=True)




# #观察数据输出结果







