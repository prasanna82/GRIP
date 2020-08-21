# Importing Libraries 
import pandas as pd
import numpy as np
iris = pd.read_csv("C:\\Users\\Prasanna\\Desktop\\Data science\\BS\\iris.csv")
iris.head
iris.dropna
len(iris)
# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(iris,test_size = 0.3) # 0.2 => 20 percent of entire data 
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC
# creating empty list variable 
acc = []
# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values  
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:5],train.iloc[:,5])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:5])==train.iloc[:,5])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:5])==test.iloc[:,5])
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt # library to do visualizations 
# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(train.iloc[:,1:5])
df_norm['clust']=wd
df_norm
df_norm.head(10)  # Top 10 rows
###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
dd['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

dd = train.iloc[:,[5,1,2,3,4]]
dd
dd.iloc[:,0:5].groupby(dd.clust).mean()




