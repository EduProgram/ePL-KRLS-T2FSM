# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:25:02 2022

@author: Eduardo Santos de Oliveira Marques
@email: eduardo.santos@engenharia.ufjf.br
"""
# Importing libraries
import math
import numpy as np
import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from time import process_time #Edu

# Calling the models
import parameters as pm


# Start to measure elapsed time
t = process_time()
    

""" Model definition stage """

# Define the model (#Edu)
model = 'ePL-KRLS-T2FSM'
    

""" Calculation stage """

# Importing the data
Data = pd.read_csv(r'C:/Users/Eduardo/Documents/GitHub/ePL-KRLS-T2FSM/Datasets/TAIEX.csv', header=0) #Edu    

# Convert the date to datetime64
Data['Date'] = pd.to_datetime(Data['Date'], format='%Y-%m-%d')
# Picking the Interval
Data_Interval = Data.loc[(Data['Date'] >= '2001-01-02') & (Data['Date'] <= '2015-12-02')]
# Picking the Close Values
Data_Type = Data_Interval['Close']
# Separating the inputs and output 
X = pd.concat([Data_Type.shift(1), Data_Type.shift(2), Data_Type.shift(3)], axis=1)
y = pd.concat([Data_Type.shift(-3)], axis=1)
# Removing the 'NaN' datas
X.dropna(subset = ['Close'], inplace=True)
y.dropna(subset = ['Close'], inplace=True)

# Changing to matrix
X = X.to_numpy()
y = y.to_numpy()

# Transforming in 1D
y = y.flatten()


""" Normalization stage """

# Normalizing the inputs
Max_X = np.max(X)
Min_X = np.min(X)
Normalized_X = (X - Min_X)/(Max_X - Min_X)

# Normalizing the output
Max_y = np.max(y)
Min_y = np.min(y)
Normalized_y = (y - Min_y)/(Max_y - Min_y);


""" Points definition and splitting stage """

# Defining the number of training points
Num_Training_Points = 3200
# Total number of points
n = Data.shape[0]

# Spliting the data into train and test
Normalized_X_train, Normalized_X_test, Normalized_y_train, Normalized_y_test = train_test_split(Normalized_X, Normalized_y, test_size = (n - Num_Training_Points)/n, shuffle = False, stratify = None)


""" Setting the model stage """

OutputTest, Rules = pm.models(model, Normalized_X_train, Normalized_y_train, Normalized_X_test, Normalized_y_test)


""" Error metrics calculation stage """

# Denormalizing the results
OutputTestDenormalized = OutputTest * (Max_y - Min_y) + Min_y
# Calculating the Error of the Coeficient of Determiation
ER_2 = 1 - r2_score(y[-300:,], OutputTestDenormalized[-300:,])
# Calculating the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y[-300:,], OutputTestDenormalized[-300:,]))
# Calculating the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y[-300:,])
# Calculating the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y[-300:,], OutputTestDenormalized[-300:,])

# End to measure elapsed time
elapsed_time = process_time() - t


""" Results """

print()
# Printing the Dataset (#Edu)
print("Dataset = TAIEX")
# Printing the model (#Edu)
print("Model = ", model) #Edu
print()
# Printing the ER^2
print("ER^2  = ", ER_2)
# Printing the NDEI
print("NDEI  = ", NDEI)
# Printing the MAPE
print("MAPE  = ", MAPE)
# Printing the number of final rules
print("Final Rules = ", Rules[-1])
print()
# Printing the elapsed time
print("Elapsed time = ", elapsed_time)

name_model = """Model name"""

# Plotting the graphic of the actual time series and its prediction
plt.plot(y[-300:,], label='Actual Value', color='red')
plt.plot(OutputTestDenormalized[-300:,], color='blue', label=name_model) #Edu
plt.suptitle('Predictions of ' + model) #Edu
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plotting rules' evolution of the model
plt.plot(Rules, color='blue')
plt.suptitle('Predictions of ' + model) #Edu
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()

# Plotting the graphic with improved resolution (comment this part if no use)
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y[-500:,], label='Actual Value', color='red')
plt.plot(OutputTestDenormalized[-500:,], color='blue', label='ePL-KRLS-T2FSM_p3 \n ePL-KRLS-T2FSM_zl \n ePL-KRLS-T2FSM_ma \n ePL-KRLS-T2FSM_hy')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper right')
plt.savefig(f'Graphics/Plots2.eps', format='eps', dpi=1200)
plt.show()