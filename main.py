import numpy as np,random as rd
import pandas as pd,csv
import os
from prework import Bag,Gram
from compassion import alpha_gradient_plot


rd.seed(2005)
np.random.seed(2004)

file_path=os.path.join('sentiment-analysis-on-movie-reviews', 'train.tsv', 'train.tsv')
with open(file_path,'r',encoding='utf-8') as f:
    train_tsv=csv.reader(f,delimiter='\t')
    tmp=list(train_tsv)

data=tmp[1:]

#特征提取
bag=Bag(data)
bag.get_matrix()

gram=Gram(data,dimension=3)
gram.get_matrix()


alpha_gradient_plot(bag,gram,1000,10)
alpha_gradient_plot(bag,gram,10000,10)







