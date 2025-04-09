import numpy as np
import random as rd

class Softmax:
    def __init__(self,sample,typenum,feature):
        self.sample=sample  #train 的样本数量
        self.typenum=typenum  #情感种类数量
        self.feature=feature  #0-1字典向量长度
        self.w=np.random.randn(feature,typenum)

    def softmax_cal(self,x):
        exp=np.exp(x-np.max(x))
        return exp/np.sum(exp)

    def softmax_all(self,wtx):
        wtx-=np.max(wtx,axis=1,keepdims=True)
        wtx=np.exp(wtx)
        wtx/=np.sum(wtx,axis=1,keepdims=True)
        return wtx

    def change_y(self,y):
        ans=np.array([0]*self.typenum)
        ans[y]=1
        return ans.reshape(-1,1)

    def prediction(self,X):    #对于0-1矩阵X,计算每个句子的y_hat的值的概率
        prob=self.softmax_all(X.dot(self.w))
        return prob.argmax(axis=1)   #argmax返回坐标，max返回值

    def correct_rate(self,train,train_y,test,test_y):
        n_train=len(train)
        pred_train=self.prediction(train)
        n_test=len(test)
        pred_test=self.prediction(test)

        train_acc=sum([train_y[i]==pred_train[i] for i in range(n_train)])/n_train
        test_acc=sum([test_y[i]==pred_test[i] for i in range(n_test)])/n_test

        print(train_acc,test_acc)
        return train_acc,test_acc

    def regression(self,X,y,alpha,epoch,strategy='mini',mini_size=100):
        if self.sample !=len(X) or self.sample !=len(y):
            raise Exception('sample and data must be same size')
        if strategy=='mini':
            for i in range(epoch):
                increment=np.zeros((self.feature,self.typenum))
                for j in range(mini_size):   #从中抽取数据j次
                    k=rd.randint(0,self.sample-1)
                    yhat=self.softmax_cal(self.w.T.dot(X[k].reshape(-1,1)))
                    increment+=X[k].reshape(-1,1).dot((self.change_y(y[k])-yhat).T)  #梯度加,交叉熵梯度=真实-预测,而Z=XW,dL/dZ=X*(真实-预测)
                self.w+=alpha/mini_size*increment

        elif strategy=='shuffle':
            for i in range(epoch):
                k=rd.randint(0,self.sample-1)
                yhat=self.softmax_cal(self.w.T.dot(X[k].reshape(-1,1)))
                increment=X[k].reshape(-1,1).dot((self.change_y(y[k])-yhat).T)
                self.w+=alpha*increment

        elif strategy=='batch':
            for i in range(epoch):
                increment=np.zeros((self.feature,self.typenum))
                for j in range(self.sample):
                    yhat=self.softmax_cal(self.w.T.dot(X[j].reshape(-1,1)))
                    increment+=X[j].reshape(-1,1).dot((self.change_y(y[j])-yhat).T)
                self.w+=alpha/self.sample*increment

        else:
            raise Exception('strategy must be mini or shuffle or batch')






