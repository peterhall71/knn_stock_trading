###https://scikit-learn.org/stable/modules/clustering.html


###Master.py

#LIBRARIES
import sys, os, shutil, statistics 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

#PARAMETERS

#exectution
Account_Value = 60000
Order_Percentage = 0.01
Trade_Images = True

#FUNCTIONS & CLASSES

class Test_Data:
    def __init__(self, name):
        self.name = name
        self.array = np.genfromtxt(os.path.join('Test_Files', name + '.csv'), delimiter=',')
        self.main_array = np.zeros((6, 4))
        self.shares = 0
        self.buy_flag = 0
        self.purchase_point = 0
        self.purchase_price = 0
        self.knn_predictions = 0
        return

    def Trade_Loop(self):
        global Account_Value
        #update main_array, select next section to insert into main array, delete last row of main_array, add new array in index 0
        self.main_array = np.delete(self.main_array, 5, axis=0)
        self.main_array = np.insert(self.main_array, 0, self.array[live_counter], axis=0)
        if live_counter < len(self.main_array): return
        
        #PREDICTION LOOP
        if self.shares + self.buy_flag == 0:
            #flatten and normalize main_array, reshape to one row np.array (1,28), make predictions, set buy_flag
            test_flat = self.main_array.flatten(order='C')
            test_norm = np.array([x/statistics.mean(test_flat) for x in test_flat])
            self.knn_predictions = knn.predict(test_norm.reshape(1,-1))
            
            if self.knn_predictions in buy_indicators:
                self.buy_flag = 1
                return
         
        #buy order
        if self.shares + self.buy_flag == 1:
            self.purchase_price = self.main_array[0][3]
            self.shares = (Order_Percentage*Account_Value)/self.purchase_price
            Account_Value = Account_Value*(1 - Order_Percentage)
            self.purchase_point = live_counter

        #SELL LOOP    
        if self.shares > 0:
                
            if live_counter - self.purchase_point == 5:
                #sell
                sell_price = self.main_array[0][3]
                Account_Value = Account_Value + self.shares*sell_price
                sell_point = live_counter
                self.buy_flag = 0

                #trading record
                profit = self.shares*(sell_price - self.purchase_price)
                time_diff = sell_point - self.purchase_point
                trade_record.append([self.purchase_point, self.purchase_price, sell_point, sell_price, time_diff, self.shares, profit, self.knn_predictions])
                
                if Trade_Images:
                    #select section to be plotted, and check if array has 11 records, the last set most likely will not
                    image_section = self.array[live_counter - 10 : live_counter + 1 ,  :]
                    if len(image_section) <11: return
                    Candlestick_Plot(image_section, self.knn_predictions, 'Trade_Images', live_counter)
                
                self.shares = 0
        return

def KNN_Classifier():
    knn = KNeighborsClassifier(
                algorithm = 'auto',
                leaf_size = 30,
                metric = 'minkowski',
                metric_params = None,
                n_jobs = None,
                n_neighbors = 6,
                p = 2,
                weights = 'uniform'
                )
    return knn

def Candlestick_Plot(plot_data, plot_title, Folder_name, file_name):
    #add static date column to array
    array_dates = np.concatenate((np.reshape(np.arange(736619,736630), (-1, 1)), plot_data), axis = 1)
    
    #configure plot, save image and close
    f1, ax = plt.subplots(figsize = (10,5))
    candlestick_ohlc(ax, array_dates, width=0.6, colorup='green', colordown='red')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.title(plot_title)
    plt.savefig(os.path.join(Folder_name, '%d.png' %file_name), bbox_inches='tight')
    plt.close()

def Input_and_Convert(message):
    while True:
        try:
            #take comma delimited input and convert to list of integers
            print('')
            cluster_list = input(message)
            cluster_list = cluster_list.split(',')
            cluster_list = [int(x.strip()) for x in cluster_list]
            break
        except:
            print('Invalid entry, please try again') 
    return cluster_list

#INITILIZATION

#turn interactive plotting off, this prevents matplotlib from dispalying all the plots, can still display with plt.show()
plt.ioff()


#EXECUTION

#prepare training data
X_train = training_data[:,0:24]
y_train = training_data[:,24]

#create and train KNN classifier
knn = KNN_Classifier()
knn.fit(X_train, y_train)

#take comma delimited input and convert to list of integers
buy_indicators = Input_and_Convert('Buy Indicators: ')

#load test datasets: X_test_1, X_test_2, etc. and initiate Main_Loop parameters
live_counter = 0
trade_record = []
Test_1 = Test_Data('X_test_1')
Test_2 = Test_Data('X_test_2')
test_array_list = [Test_1, Test_2]

while live_counter < len(Test_1.array) - 10:
    
    #increment live_counter
    live_counter += 1
    
    for each in test_array_list:
        each.Trade_Loop()

np.savetxt("tradingRecord.csv", np.array(trade_record), delimiter=",")
print('Number of Trades:', len(trade_record))
print('Ending Account Value:', Account_Value)

