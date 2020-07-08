# 3_KMeans.py

# load libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# global parameters
plt.ioff() # plots will not show unless explicitly called

# set working directory
os.chdir(r'C:\Users\Peter A Hall\Documents\GitHub\knn_stock_trading')

# load prepared data
prepedData = pd.read_csv('preparedData.csv', header = 0)

# create K-Means classifier
km = KMeans(
        n_clusters = 15,
        init = 'k-means++',
        n_init = 10,
        max_iter = 300,
        tol = 1e-04,
        precompute_distances = 'auto',
        verbose = 0,
        random_state=None,
        copy_x = True,
        n_jobs = -1,
        algorithm = 'auto'
)

# fit data using KMeans
km.fit(prepedData)

# add cluster labels to data
labels = pd.DataFrame(km.labels_, columns = ['cluster'])
clusteredData = pd.concat([prepedData, labels], axis = 1)

# create set of cluster labels
clusterList = labels.cluster.unique()
clusterList.sort()

# create images of averaged clusters
for each in clusterList:
    # create df for specific cluster based on each
    # remove cluster column, print number of records in cluster
    subCluster = clusteredData[clusteredData['cluster'] == each]
    del subCluster['cluster']
    print('Record Count,' , each,':', len(subCluster))
    
    # find mean of columns
    # reformat avgCluster from one averaged 44 column row to set of 11 ohlc records
    avgCluster = subCluster.mean(axis=0)
    avgOHLC = pd.DataFrame(avgCluster.values.reshape(11,4), columns = ['Open', 'High', 'Low', 'Close'])

    # change index to datetimeindex with static dates
    avgOHLC.set_index(keys = np.arange(736619,736630), inplace=True)
    avgOHLC.index = pd.to_datetime(avgOHLC.index)
    avgOHLC.index.name = 'Date'
    
    # create plot and save
    mpf.plot(avgOHLC, 
            type = 'candle', 
            style = 'charles',
            title = '%d' %each,
            ylabel = 'Normalized Price', 
            volume = False, 
            savefig = 'Cluster_Images\%d.png' %each
    )   

# remove prediction portion of the array
clusteredData.drop(clusteredData.columns[24:44], axis = 1, inplace = True)

# save training_data to csv
clusteredData.to_csv('trainingData.csv', index = False)

# split dataset into training data and clusters
X_analysis = clusteredData.iloc[:,0:24]
Y_analysis = clusteredData.iloc[:,24]

# split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X_analysis, Y_analysis, test_size = 0.2, random_state = None)

# create and train KNN classifier, make predictions on unseen test data
knn = KNeighborsClassifier(
        algorithm = 'auto',
        leaf_size = 30,
        metric = 'minkowski',
        metric_params = None,
        n_jobs = -1,
        n_neighbors = 6,
        p = 2,
        weights = 'uniform'
)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# returns the coefficient of determination R^2 of the prediction and confusion matrix
knn.score(X_test, y_test)
confusion_matrix(y_test, knn_predictions, clusterList)
