import numpy as np
import random as rd

mx=2000

#将训练集划分为训练集+测试集合
def data_split(data):
    train=list()
    test=list()
    rate=0.3
    num=0
    l=len(data)
    for it in data:
        num+=1
        if rd.random()>rate:
            train.append(it)
        else:
            test.append(it)
        if num==l or num==mx:
            break
    return train,test

#定义词袋类
class Bag:
    def __init__(self,data):
        self.data=data[:mx]
        self.max_item=mx
        self.dict_word=dict()
        self.train,self.test=data_split(data)
        self.len=len(self.dict_word)
        self.train_y=[int(it[3]) for it in self.train]
        self.test_y=[int(it[3]) for it in self.test]#记录情绪
        self.train_matrix=None
        self.test_matrix=None  #0-1矩阵，一行一句

    def get_matrix(self):
        for it in self.data:
            s=it[2]
            s=s.upper()
            s_split=s.split() #按照空格分割
            for word in s_split:
                if word not in self.dict_word:
                    self.dict_word[word]=len(self.dict_word)#对每一个单词映射编号
        self.len=len(self.dict_word)
        self.train_matrix=np.zeros((len(self.train),self.len))
        self.test_matrix=np.zeros((len(self.test),self.len))

        for i in range(len(self.train)):
            s=self.train[i][2]
            s=s.upper()
            s_split=s.split()
            for word in s_split:
                if word  in self.dict_word:
                    self.train_matrix[i][self.dict_word[word]]=1

        for i in range(len(self.test)):
            s=self.test[i][2]
            s=s.upper()
            s_split=s.split()
            for word in s_split:
                if word  in self.dict_word:
                    self.test_matrix[i][self.dict_word[word]]=1

#定义N元特征
class Gram:
    def __init__(self,data,dimension=2):
        self.data=data[:mx]
        self.dimension=dimension
        self.max_item=mx
        self.dict_word=dict()
        self.len=0
        self.train,self.test=data_split(data)
        self.train_y=[int(it[3]) for it in self.train]
        self.test_y=[int(it[3]) for it in self.test]
        self.train_matrix=None
        self.test_matrix=None

    def get_matrix(self):
        for d in range(1,self.dimension+1):
            for it in self.data:
                s=it[2]
                s=s.upper()
                s_split=s.split()
                l=len(s_split)
                for i in range(l-d+1):
                    tmp=s_split[i:i+d]
                    tmp='_'.join(tmp)
                    if tmp not in self.dict_word:
                        self.dict_word[tmp]=len(self.dict_word)
        self.len=len(self.dict_word)
        self.train_matrix=np.zeros((len(self.train),self.len))
        self.test_matrix=np.zeros((len(self.test),self.len))

        for d in range(1,self.dimension+1):
            for i in range(len(self.train)):
                s=self.train[i][2]
                s=s.upper()
                s_split=s.split()
                for j in range(len(s_split)-d+1):
                    tmp=s_split[j:j+d]
                    tmp='_'.join(tmp)
                    if tmp in self.dict_word:
                        self.train_matrix[i][self.dict_word[tmp]]=1

            for i in range(len(self.test)):
                s=self.test[i][2]
                s=s.upper()
                s_split=s.split()
                for j in range(len(s_split)-d+1):
                    tmp=s_split[j:j+d]
                    tmp='_'.join(tmp)
                    if tmp in self.dict_word:
                        self.test_matrix[i][self.dict_word[tmp]]=1





























