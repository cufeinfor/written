# import word_sequence
# from model import tokenlize

# from model import tokenlize
from dataset import *
# from word_sequence import *
from word_sequence import Word2Sequence
# import tokenlize
# import word_sequence
import os
import pickle
from tqdm import tqdm



if __name__=='__main__': 
    word=Word2Sequence()
    path=r"C:/Users/qiaoy/Desktop/data/aclImdb/train"

    #把所有文件名放到列表里
    tem_data_path=[os.path.join(path,"pos"),os.path.join(path,"neg")]

    for data_path in tem_data_path:
        file_path_list=[os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith("txt")]
        for file_path in tqdm(file_path_list):
            sentence=Mytokenize(open(file_path,encoding='UTF-8').read())
            word.fit(sentence)
    word.build_vocab(min=10)
    pickle.dump(word,open("C:/Users/qiaoy/Desktop/written/model/word.pkl","wb"))
    # print(word.dict)
    print(len(word))
    # print(lib.ws)
