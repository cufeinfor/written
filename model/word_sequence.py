#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@Description:       :
@Date     :2021/11/17 20:40:52
@Author      :Amanda
@version      :1.0
'''

#实现构建词典，实现方法把句子转化为数字序列和其翻转


class Word2Sequence():
#    原始词典信息
    UNK_TAG="UNK"#特殊字符（没有出现在词典里的
    PAD_TAG="PAD"#短句子填充成相同的长度

    UNK=0
    PAD=1
#原始词典信息，初始化
    def __init__(self):
        self.dict={
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count={}#统计词频

    #传入数据
    def fit(self,sentence):
        """
        @description  :把单个句子保存到dict中
        ---------
        @param  :sentence:[word1,word2.....]列表
        -------
        @Returns  :
        -------
        """
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1#如果没有那个词，则+1，即只有一个

    def build_vocab(self,min=5,max=None,max_features=None):
        """
        @description  :生成词典
        ---------
        @param  :min:最小出现的次数，max:最大出现的次数，max_features：一共保留多少个词语
        -------
        @Returns  :
        -------
        """
        # 删除count中词频小于min的词语
        if min is not None:
           self.count={word:value for word,value in self.count.items() if value>min}
        # 删除count中词频大于max的词语
        if max is not None:
           self.count={word:value for word,value in self.count.items() if value<max}
        #限制保留的词语数量,根据词典进行排序，保存前多少个
        if max_features is not None:
            temp=sorted(self.count.items(),key=lambda x:x[-1],reversed=True)[:max_features]
            self.count=dict(temp)
        # 转换成数字序列，用len表示
        for word in self.count:
            self.dict[word]=len(self.dict)
        # 得到一个翻转的Dict字典
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))


    def transform(self,sentence,max_len=None):
        """
        @description  :把文本转成数字序列
        ---------
        @param  :sentence:[word1,word2.....]列表
        @param:max_len:对句子进行填充或者裁减
        -------
        @Returns  :
        -------
        """

        # for word in sentence:
        #     self.dict.get(word,self.UNK)
        # $填充
        if max_len is not None and max_len>len(sentence):
            sentence=sentence+[self.PAD_TAG]*(max_len-len(sentence))
        # 裁减
        if max_len is not None and max_len<len(sentence):
            sentence=sentence[:max_len]
        return[self.dict.get(word,self.UNK) for word in sentence]


    def inverse_transform(self,indices):
        """
        @description  :把数字序列转换成句子文本
        ---------
        @param  :心底测试：【1.2.3.4】
        -------
        @Returns  :
        -------
        """
        
        return[self.inverse_dict.get(idx) for idx in indices]
    def __len__(self):
        return len(self.dict)




# if __name__=='__main__': 


#     from os import path
#     from typing import Dict
#     import numpy as np
#     # from ..model import word_sequence
#     import pickle #用pickle来序列化
#     import os
#     # from ..model import tokenlize
#     # from ..import word_sequnce
#     from ..import tokenlize


#     word=Word2Sequence()
#     path=r"C:/Users/qiaoy/Desktop/data/aclImdb/train"

#     #把所有文件名放到列表里
#     tem_data_path=[os.path.join(path,"pos"),os.path.join(path,"neg")]

#     for data_path in tem_data_path:
#         file_path_list=[os.path.join(data_path,file_name) for file_name in os.listdir(data_path)]
#         # file_path=[os.path.join(data_path,file_name) for file_name in os.path.join(data_path,file_name)]
#         for file_path in file_path_list:
#             sentence=tokenlize.Mytokenize(open(file_path).read())
#             word.fit(sentence)
#     word.build_vocab(min=10)
#     pickle.dump(word,open("./model/word.pkl","rb"))
#     print(word.dict)
#     print(len(word))

    

            
        


        


   
