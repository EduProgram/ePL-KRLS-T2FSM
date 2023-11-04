# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:55:26 2021

@author: Eduardo Santos de Oliveira Marques
@email: eduardo.santos@engenharia.ufjf.br
"""

# Importing libraries
import math
import numpy as np
import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from time import process_time 

# Calling the models
import parameters as pm


# Start to measure elapsed time
t = process_time()


""" Model definition stage """

# Define the model
model = 'ePL-KRLS-T2FSM'


""" Calculation stage """

# Importing the data (change the path)
Data = pd.read_excel (r'C:/Users/Eduardo/Documents/GitHub/ePL-KRLS-T2FSM/Datasets/MackeyGlass.xlsx', header=None)

# Changing to matrix
Data = Data.to_numpy()
# Picking the data from the matrix
x = Data[:,:]
# Transforming in 1D
x = x.flatten()

# Defining the parameters for Mackey-Glass equation
a   = 10  # Initial value: 10
b   = 0.1 # Initial value: 0.1
c   = 0.2 # Initial value: 0.2
tau = 1000 # Initial value: 17, 500, 1000

# Setting the first value
Y = x[0]
# Setting the input to the time series (according to 'tau')
for j in range(1, tau + 1):
    Y = np.append(Y, x[j])

# Generating a chaotic time series with Mackey-Glass equation
for i in range(tau, len(x) + 99):
    Y = np.append(Y, Y[i] - b*Y[i] + c*Y[i-tau]/(1+Y[i-tau]**a)) 
Y = Y[100:]

# Reshaping the time series according to the datasheet
Y = Y.reshape(len(Data[:,:-1]), len(Data[-1,:]))
# Separating the inputs and output
X = Y[:,:-1]
y = Y[:,-1]


""" Normalization stage """

# Normalizing the inputs
Max_X = np.max(X)
Min_X = np.min(X)
Normalized_X = (X - Min_X)/(Max_X - Min_X)

# Normalizing the output
Max_y = np.max(y)
Min_y = np.min(y)
Normalized_y = (y - Min_y)/(Max_y - Min_y)


""" Points definition and splitting stage """

# Defining the number of training points
Num_Training_Points = 3000 # Initial value: 3000
# Total number of points
n = Data.shape[0]

# Spliting the data into train and test
Normalized_X_train, Normalized_X_test, Normalized_y_train, Normalized_y_test = train_test_split(Normalized_X, Normalized_y, test_size = (n - Num_Training_Points)/n, shuffle = False, stratify = None)


""" Setting the model stage """

OutputTest, Rules = pm.models(model, Normalized_X_train, Normalized_y_train, Normalized_X_test, Normalized_y_test)

    
""" Error metrics calculation stage """

# Denormalizing the results
OutputTestDenormalized = OutputTest * (Max_y - Min_y) + Min_y
# Calculating the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y[-500:,], OutputTestDenormalized[-500:,]))
# Calculating the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y[-500:,])
# Calculating the Mean Absolute Error
MAE = mean_absolute_error(y[-500:,], OutputTestDenormalized[-500:,])


# End to measure elapsed time
elapsed_time = process_time() - t


""" Results """

print()
# Printing the Datase
print("Dataset = MackeyGlass with Delay =", tau)
# Printing the model
print("Model = ", model) 
# Printing the measure (if the model's ePL-KRLS-SDM)
# if model == 'ePL-KRLS-FSM':
#        print("Measure = ", measure)
print()
# Printing the RMSE
print("RMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE  = ", MAE)
# Printing the number of final rules
print("Final Rules = ", Rules[-1])
print()
# Printing the elapsed time
print("Elapsed time = ", elapsed_time)


name_model = """Model name"""

# Plotting the graphic of the actual time series and its prediction
plt.plot(y[-500:,], label='Actual Value', color='red')
plt.plot(OutputTestDenormalized[-500:,], color='blue', label=name_model) 
plt.suptitle('Predictions of ' + model) 
plt.title('Results with tau = ' + str(tau)) 
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend()
plt.show()

# Plotting rules' evolution of the model
plt.plot(Rules, color='blue')
plt.suptitle('Predictions of ' + model)
plt.title('Results with tau = ' + str(tau))
plt.ylabel('Number of Fuzzy Rules')
plt.xlabel('Samples')
plt.show()

# Plotting the graphic with improved resolution (comment this part if no use)
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y[-500:,], label='Actual Value', color='red')
plt.plot(OutputTestDenormalized[-500:,], color='blue', label='ePL-KRLS-FSM_r2')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper right')
plt.savefig(f'Graphics/Plots2.eps', format='eps', dpi=1200)
plt.show()