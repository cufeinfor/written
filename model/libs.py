import pickle
import torch.nn as nn
model_loader=pickle.load(open("C:/Users/qiaoy/Desktop/written/model/word.pkl","rb"))
# # model_loader=pickle.load(open("word.pkl","rb"))

Max_len=20

# print(len(model_loader))
# print(nn.Embedding(len(model_loader),100))
