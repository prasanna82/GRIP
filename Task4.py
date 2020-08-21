import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\Prasanna\\Desktop\\Data science\\BS\\iris.csv")
data.head()
data['Species'].unique()
data.Species.value_counts()
data.Species
colnames = list(data.columns)
predictors = colnames[1:5]
predictors
target = colnames[5]
target
# Splitting data into training and testing data set

import numpy as np
data
# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
data['is_train'] = np.random.uniform(0, 1, len(data))<= 0.75
data['is_train']
np.random.uniform(0, 1, len(data))
train,test = data[data['is_train'] == True],data[data['is_train']==False]
train
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
train
model = DecisionTreeClassifier(criterion ='gini')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
# Accuracy = train
np.mean(train.Species == model.predict(train[predictors]))
# Accuracy = Test
np.mean(preds==test.Species) # 1




