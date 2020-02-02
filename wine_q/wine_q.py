import numpy as np
import pandas as pd
import os, sys
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset_file = pd.read_csv('wine_red.csv', sep = ';')
dataset_file = dataset_file.drop(['fixed acidity'], 1)
dataset_file = dataset_file.drop(['volatile acidity'],1)
dataset_file = dataset_file.drop(['citric acid'],1)
dataset_file = dataset_file.drop(['residual sugar'],1)
dataset_file = dataset_file.drop(['chlorides'],1)
dataset_file = dataset_file.drop(['free sulfur dioxide'],1)
dataset_file = dataset_file.drop(['total sulfur dioxide'],1)

dataset_file = dataset_file.drop(['density'],1)
dataset_file = dataset_file.drop(['pH'],1)
dataset_file = dataset_file.drop(['sulphates'],1)
dataset_file = dataset_file.drop(['alcohol'],1)
print(dataset_file)
values = DataFrame(dataset_file.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.colimns = ['t-1', 't+1']

X = dataframe.values
train_size = int(len(X)*0.50)
train, test = X[1:train_size], X[train_size:]
train_x, train_y = train[:,0], train[:,1]
test_x, test_y = test[:,0],test[:,1]

def model_pers(x):
    return x

prediction = list()
for x in test_x:
    y_hat = model_pers(x)
    prediction.append(y_hat)

pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in prediction])
dataset_file.plot()
pyplot.show()