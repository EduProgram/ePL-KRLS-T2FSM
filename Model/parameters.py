# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:59:35 2021

@author: Eduardo Santos de Oliveira Marques
@email: eduardo.santos@engenharia.ufjf.br
"""

# Importing the model
from ePL_KRLS_T2FSM import ePL_KRLS_T2FSM

def models(model, X_train, y_train, X_test, y_test):
    
    if model == 'ePL-KRLS-T2FSM':
        
        " Metric definition "
    
        # Defining the measure
        measure = 'ralescu2'
            # Options in State-of-the-art: 'ePL-KRLS', 'ePL-KRLS-DISCO'
            # Options in Similarity T1: 'pappis1', 'pappis2', 'pappis3', 'jaccard', 
                                      # 'dice', 'zwick', 'chen', 'vector'
            
            # Options in Distance T1: 'ralescu1', 'ralescu2', 'chaudhuri_rosenfeld', 
                                    # 'chaudhuri_rosenfeld_nn', 'grzegorzewski_non_inf_pq', 
                                    # 'grzegorzewski_non_inf_p', 'grzegorzewski_inf_q', 
                                    # 'grzegorzewski_inf', 'ban', 'allahviranloo', 
                                    # 'yao_wu', 'mcculloch'
            
            # Options in Incompatibility measure: 'compatibility'
            
            # Options in Similarity GT2: 'jaccard_gt2', 'zhao_crisp', 'hao_fuzzy', 
                                       # 'hao_crisp', 'yang_lin', 'mohamed_abdaala', 
                                       # 'hung_yang', 'wu_mendel'
                                       
            # Options in Similarity IT2: 'zeng_li', 'gorzalczany', 'bustince', 
                                       # 'jaccard_it2', 'zheng', 'vector_it2'
                                       
            # Options in Distance GT2: 'mcculloch_gt2'
            
            # Options in Distance IT2: 'figueroa_garcia_alpha', 
                                     # 'figueroa_garcia_centres_hausdorff', 
                                     # 'figueroa_garcia_centres_minkowski', mcculloch_it2
            
    
        # Defining the membership function (if is Similarity or Distance metric)
        mfA = 'gaussian_t1'
        mfB = 'gaussian_t1'
            # Options in Membership Function T1: 'gaussian_t1', 'polling_t1', 'iaa_t1', 'discrete_t1'
            # Options in Membership Function T2: 'gaussian_t2', 'polling_t2', 'iaa_t2'
        
        # Printing the measure
        print("Measure = ", measure)
        # Printing the MFs
        print("MF of A = ", mfA)
        # Printing the MFs
        print("MF of B = ", mfB)
        
        " Model hyperparameters stage "
        
        # Setting the hyperparameters
        alpha = 0.001 # MG: 0.001 ; ST: 0.01                                    Interval <= 1
        beta = 0.06 # MG: 0.06 ; ST: 0.1                                        Interval = [0, 1]
        lambda1 = 10**(-7) # MG: 10**(-7) ; ST 10**(-3)                        Interval <= 1
        sigma = 0.3 # MG: 0.3 ; ST: 0.5                                        Interval = [0.2, 0.5]
        omega = 1 # MG: 1 ; ST: 1                                              Interval = 1
        e_utility = 0.05 # MG: 0.05 ; ST: 0.05                                 Interval = [0.03, 0.05]
        
        " Initializing, training and testing stage "
        
        # Initializing the model
        Model = ePL_KRLS_T2FSM(alpha = alpha, beta = beta, lambda1 = lambda1, sigma = sigma, tau = beta, e_utility = e_utility, omega = omega, measure = measure, mfA = mfA, mfB = mfB) #Edu
        # Training the model
        OutputTraining, Rules = Model.Train(X_train, y_train, measure, mfA, mfB) #Edu
        # Testing the model
        OutputTest = Model.Test(X_test, y_test, measure, mfA, mfB) #Edu
        
        return OutputTest, Rules
    