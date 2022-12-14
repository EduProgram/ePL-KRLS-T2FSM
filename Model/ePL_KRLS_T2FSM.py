# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Eduardo Santos de Oliveira Marques
@email: eduardo.santos@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np
import math
import measures as m
import fuzzy_sets as gfs
from decimal import Decimal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # pandas (append -> concat)


class ePL_KRLS_T2FSM:    
    def __init__(self, alpha = 0.001, beta = 0.05, lambda1 = 0.0000001, sigma = 0.5, tau = 0.05, omega = 1, e_utility = 0.05, measure = 'pappis1', mfA = 'gaussian_t1', mfB = 'gaussian_t1'):
        self.hyperparameters = pd.DataFrame({'alpha':[alpha],'beta':[beta], 'lambda1':[lambda1], 'sigma':[sigma], 'tau':[tau], 'omega':[omega], 'e_utility':[e_utility]})
        self.parameters = pd.DataFrame(columns = ['Center', 'Dictionary', 'nu', 'P', 'Q', 'Theta','ArousalIndex', 'Utility', 'SumLambda', 'TimeCreation', 'CompatibilityMeasure'])
        # Parameters used to calculate the utility measure
        self.epsilon = []
        self.eTil = [0.]
        # Monitoring if some rule was excluded
        self.ExcludedRule = 0
        # Evolution of the model rules
        self.rules = []
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
    
        
    def Train(self, X, y, measure, mfA, mfB):
        # Initialize the first rule
        self.parameters = self.parameters.append(self.Initialize_First_Cluster(X[0,], y[0]), ignore_index = True)
        for k in range(1, X.shape[0]):
            # Compute the compatibility measure and the arousal index for all rules
            if k == 235:
                hjkh = 0
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i, measure, mfA, mfB)
                self.Arousal_Index(i)
            # Find the minimum arousal index
            MinIndexArousal = self.parameters['ArousalIndex'].idxmin()
            # Find the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].idxmax()
            # Verifying the needing to creating a new rule
            if self.parameters.loc[MinIndexArousal, 'ArousalIndex'] > self.hyperparameters.loc[0, 'tau'] and self.ExcludedRule == 0:
                self.parameters = self.parameters.append(self.Initialize_Cluster(X[k,], y[k], k+1, MaxIndexCompatibility), ignore_index = True)
            else:
                self.Rule_Update(X[k,], y[k], MaxIndexCompatibility)
            self.Updating_Lambda(X[k,])
            if self.parameters.shape[0] > 1:
                self.Utility_Measure(X[k,], k+1)
            self.rules.append(self.parameters.shape[0])
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].idxmax()
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'].shape[0]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'Theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'][ni,], X[k,])
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y[k])**2)
            # Updating epsilon and e_til
            self.epsilon.append(math.exp(-0.5) * (2/(math.exp(-0.8 * self.eTil[-1] - abs(Output - y[k]))) - 1))
            self.eTil.append(0.8 * self.eTil[-1] + abs(Output - y[k]))
        return self.OutputTrainingPhase, self.rules
      
      
    def Test(self, X, y, measure, mfA, mfB):
        for k in range(X.shape[0]):
            # Computing the compatibility measure
            for i in self.parameters.index:
                self.Compatibility_Measure(X[k,], i, measure, mfA, mfB)
            # Finding the maximum compatibility measure
            MaxIndexCompatibility = self.parameters['CompatibilityMeasure'].idxmax()
            # Computing the output
            Output = 0
            for ni in range(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'].shape[0]):
                Output = Output + self.parameters.loc[MaxIndexCompatibility, 'Theta'][ni] * self.Kernel_Gaussiano(self.parameters.loc[MaxIndexCompatibility, 'Dictionary'][ni,], X[k,])
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output)
            self.ResidualTestPhase = np.append(self.ResidualTestPhase, (Output - y[k])**2)
        return self.OutputTestPhase
    
        
    def Initialize_First_Cluster(self, x, y):
        Q = np.linalg.inv(np.ones((1,1))*(self.hyperparameters.loc[0, 'lambda1'] + (self.Kernel_Gaussiano(x, x))))
        Theta = Q*y
        NewRow = {'Center': x.reshape((1,len(x))), 'Dictionary': x.reshape((1,len(x))), 'nu': self.hyperparameters['sigma'][0], 'P': np.ones((1,1)), 'Q': Q, 'Theta': Theta, 'ArousalIndex': 0., 'Utility': 1., 'SumLambda': 0., 'NumObservations': 1., 'TimeCreation': 1., 'CompatibilityMeasure': 1.}
        Output = y
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(Output - y)**2)
        return NewRow
    
    
    def Initialize_Cluster(self, x, y, k, i):
        Q = np.linalg.inv(np.ones((1,1))*(self.hyperparameters.loc[0, 'lambda1'] + (self.Kernel_Gaussiano(x, x))))
        Theta = Q*y
        nu = (np.linalg.norm(x - self.parameters.loc[i, 'Center'])/math.sqrt(-2 * np.log(max(self.epsilon))))
        NewRow = {'Center': x.reshape((1,len(x))), 'Dictionary': x.reshape((1,len(x))), 'nu': nu, 'P': np.ones((1,1)), 'Q': Q, 'Theta': Theta, 'ArousalIndex': 0., 'Utility': 1., 'SumLambda': 0., 'NumObservations': 1., 'TimeCreation': k, 'CompatibilityMeasure': 1.}
        return NewRow
    
    
    def Kernel_Gaussiano(self, Vector1, Vector2):
        return math.exp(-((((np.linalg.norm(Vector1-Vector2))**2)/(2*self.hyperparameters.loc[0, 'sigma']**2))))
    
    
    def Fuzzy_Set(self, x, mf, i):
        # print(x)
        # If the data is 2D NumPy array, convert to a 1D
        if x.shape != x[0].shape:
            x = x.flatten()
        # Transform the NumPy array in a list
        d = list(x)
        # Constructing membership values for T1
        if mf == 'gaussian_t1':
            return gfs.generate_gaussian_t1_fuzzy_set(x)
        if mf == 'polling_t1':
            return gfs.generate_polling_t1_fuzzy_set(d)
        if mf == 'iaa_t1':
            return gfs.generate_iaa_t1_fuzzy_set(d)
        if mf == 'discrete_t1':
            return gfs.generate_discrete_t1_fuzzy_set(d)
        
        # Constructing membership values for T2 
        if mf == 'gaussian_t2' or 'polling_t2' or 'iaa_t2':
            d_subsets = np.array_split(d, 2)
            d1 = list(d_subsets[0])
            d2 = list(d_subsets[1])
            # If the list is unpaired, repeat the midpoint in the second subset
            if (len(d)%2) != 0:
                middle = len(d)/2
                point_middle = math.ceil(middle) - 1
                d2.insert(0, d[point_middle])
            # print(d1)
            # print(d2)
            if mf == 'gaussian_t2':
                return gfs.generate_gaussian_t2_fuzzy_set((d1, d2))
            if mf == 'polling_t2':
                return gfs.generate_polling_t2_fuzzy_set((d1, d2))
            if mf == 'iaa_t2':
                return gfs.generate_iaa_t2_fuzzy_set((d1, d2))
            
            # if mf == 'polling_t2' or 'iaa_t2':
            # mean = np.mean(d)
            # d1 = (mean, 1, len(d))
            # d2 = (mean, 0.5, len(d))
            # d3 = (mean, 0.25, len(d))
        
        
    def Measures(self, x, i, measure, mfA, mfB):
        # Cluster center
        v = self.parameters.loc[i, 'Center']
        # Membership function A
        A = self.Fuzzy_Set(x, mfA, i)
        # Membership function B
        B = self.Fuzzy_Set(v, mfB, i)
       
        #--------------------------------------------------------------------------

        # Original Code measures
        
        if measure == 'ePL-KRLS':
            return (m.ePL_KRLS(x, v))
        
        if measure == 'ePL-KRLS-DISCO':
            return m.ePL_KRLS_DISCO(x, v)

        #--------------------------------------------------------------------------

        # Similarity measures (Type-1)
        
        if measure == 'pappis1':
             "Based on the maximum distance between membership values."
             return float(m.pappis1(A, B))
         
        if measure == 'pappis2':
            "The ratio between the negation and addition of membership values."
            return float(m.pappis2(A, B))
        
        if measure == 'pappis3':
            "Based on the average difference between membership values."
            return float(m.pappis3(A, B))
        
        if measure == 'jaccard':
            "Ratio between the intersection and union of the fuzzy sets."
            return float(m.jaccard(A, B))
        
        if measure == 'dice':
            "Based on the ratio between the intersection and cardinality."
            return float(m.dice(A, B))
        
        if measure == 'zwick':
            "The maximum membership of the intersection of the fuzzy sets."
            return float(m.zwick(A, B))
        
        if measure == 'chen':
            "Ratio between the product of memberships and the cardinality."
            return float(m.chen(A, B))
        
        if measure == 'vector':
            "Vector similarity based on the distance and similarity of shapes."
            return float(m.vector(A, B))

        #--------------------------------------------------------------------------

        # Distance measures (Type-1)
        
        if measure == 'ralescu1':
             "Calculate the average Hausdorff distance over all alpha-cuts."
             return float(1 - m.ralescu1(A, B))
         
        if measure == 'ralescu2':
             "Calculate the maximum Hausdorff distance over all alpha-cuts."
             return float(1 - m.ralescu2(A, B))
         
        if measure == 'chaudhuri_rosenfeld':
             "Calculate the weighted average of Hausdorff distances."
             return float(1 - m.chaudhuri_rosenfeld(A, B))
         
        if measure == 'chaudhuri_rosenfeld_nn':
             "Calculate the weighted average of Hausdorff distances for non-normal."
             return float(1 - m.chaudhuri_rosenfeld_nn(A, B))
         
        if measure == 'grzegorzewski_non_inf_pq':
             "Grzegorzewski distance where 1 <= p < infty and q is used."
             return float(1 - m.grzegorzewski_non_inf_pq(A, B))
         
        if measure == 'grzegorzewski_non_inf_p':
             "Grzegorzewski distance where 1 <= p < infty and q is not used."
             return float(1 - m.grzegorzewski_non_inf_p(A, B))
         
        if measure == 'grzegorzewski_inf_q':
             "Grzegorzewski distance where p is infinity and q is used."
             return float(1 - m.grzegorzewski_inf_q(A, B))
         
        if measure == 'grzegorzewski_inf':
             "Grzegorzewski distance where p is infinity and q is not used."
             return float(1 - m.grzegorzewski_inf(A, B))
         
        if measure == 'ban':
             "Minkowski based distance."
             return float(1 - m.ban(A, B))
         
        if measure == 'allahviranloo':
             "Distance based on the average width and centre of the fuzzy sets."
             return float(1 - m.allahviranloo(A, B))
         
        if measure == 'yao_wu':
             "Calculate the average Minkowski (r=1) distance."
             return float(1 - m.yao_wu(A, B))
         
        if measure == 'mcculloch':
             "Calculate the weighted Minkowski (r=1) directional distance."
             return float(1 - m.mcculloch(A, B))
         
        #--------------------------------------------------------------------------

        # Incompatibility Measure (Type-1)
        
        if measure == 'compatibility':
             "Calculate weighted average of dissimilarity and directional distance."
             return float(m.compatibility(A, B))
         
        
        #--------------------------------------------------------------------------
        #--------------------------------------------------------------------------
        
        # Similarity measures (General Type-2)
        
        if measure == 'jaccard_gt2':
            "Calculate the weighted average of the jaccard similarity on zslices."
            return float(m.jaccard_gt2(A, B))
        
        if measure == 'zhao_crisp':
            "Like jaccard, but the result is the standard average; not weighted."
            return float(m.zhao_crisp(A, B))
        
        if measure == 'hao_fuzzy':
            "Calculate the jaccard similarity given as type-1 fuzzy set."
            return float(m.hao_fuzzy(A, B))
        
        if measure == 'hao_crisp':
            "Calculate the centroid of hao_fuzzy(fs1, fs2)."
            return float(m.hao_crisp(A, B))
        
        if measure == 'yang_lin':
            "Calculate the average jaccard similarity for each vertical slice."
            return float(m.yang_lin(A, B))
        
        if measure == 'mohamed_abdaala':
            "Based on the the jaccard similarity for each vertical slice."
            return float(m.mohamed_abdaala(A, B))
        
        if measure == 'hung_yang':
            "Based on the Hausdorff distance between vertical slice pairs."
            return float(m.hung_yang(A, B))
        
        if measure == 'wu_mendel':
            "Geometric approach."
            return float(m.wu_mendel(A, B))
        
        #--------------------------------------------------------------------------
        
        # Similarity measures (Interval Type-2)
        
        if measure == 'zeng_li':
            "Based on the average distance between the membership values."
            return float(m.zeng_li(A, B))
        
        if measure == 'gorzalczany':
            "Based on the highest membership where the fuzzy sets overlap."
            return float(m.gorzalczany(A, B))
        
        if measure == 'bustince':
            "Based on the inclusion of one fuzzy set within the other."
            return float(m.bustince(A, B))
        
        if measure == 'jaccard_it2':
            "Ratio between the intersection and union of the fuzzy sets."
            return float(m.jaccard_it2(A, B))
        
        if measure == 'zheng':
            "Similar to jaccard; based on the intersection and union of the sets."
            return float(m.zheng(A, B))
        
        if measure == 'vector_it2':
            "Vector similarity based on the distance and similarity of shapes."
            return float(m.vector_it2(A, B))
        
        #--------------------------------------------------------------------------
        
        # Distance measures (General Type-2)
        
        if measure == 'mcculloch_gt2':
            "Calculate the weighted Minkowski (r=1) directional distance."
            return float(1 - m.mcculloch_gt2(A, B))
        
        #--------------------------------------------------------------------------
        
        # Distance measures (Interval Type-2)
        
        if measure == 'figueroa_garcia_alpha':
            "Calculate the absolute difference between alpha-cuts."
            return float(1 - m.figueroa_garcia_alpha(A, B))
        
        if measure == 'figueroa_garcia_centres_hausdorff':
            "Calculate the hausdorff distance between the centre-of-sets."
            return float(1 - m.figueroa_garcia_centres_hausdorff(A, B))
        
        if measure == 'figueroa_garcia_centres_minkowski':
            "Calculate the absolute difference between the centre-of-sets."
            return float(1 - m.figueroa_garcia_centres_minkowski(A, B))
        
        if measure == 'mcculloch_it2':
            "Calculate the weighted Minkowski (r=1) directional distance."
            return float(1 - m.mcculloch_it2(A, B))
        
        #--------------------------------------------------------------------------


    def Compatibility_Measure(self, x, i, measure, mfA, mfB):
        self.parameters.at[i, 'CompatibilityMeasure'] = self.Measures(x, i, measure, mfA, mfB)
        
        
    def Arousal_Index(self, i):
        self.parameters.at[i, 'ArousalIndex'] = self.parameters.loc[i, 'ArousalIndex'] + self.hyperparameters.loc[0, 'beta']*(1 - self.parameters.loc[i, 'CompatibilityMeasure'] - self.parameters.loc[i, 'ArousalIndex']) #NewEdu
    
    
    def Rule_Update(self, x, y, i):
        # Update the number of observations in the rule
        self.parameters.loc[i, 'NumObservations'] = self.parameters.loc[i, 'NumObservations'] + 1
        # Store the old cluster center
        OldCenter = self.parameters.loc[i, 'Center']
        # Update the cluster center
        self.parameters.at[i, 'Center'] = self.parameters.loc[i, 'Center'] + (self.hyperparameters.loc[0, 'alpha']*(self.parameters.loc[i, 'CompatibilityMeasure'])**(1 - self.parameters.loc[i, 'ArousalIndex']))*(x - self.parameters.loc[i, 'Center'])
        # Update the kernel size
        self.parameters.at[i, 'nu'] = math.sqrt((self.parameters.loc[i, 'nu'])**2 + (((np.linalg.norm(x - self.parameters.loc[i, 'Center']))**2 - (self.parameters.loc[i, 'nu'])**2)/self.parameters.loc[i, 'NumObservations']) + ((self.parameters.loc[i, 'NumObservations'] - 1) * ((np.linalg.norm(self.parameters.loc[i, 'Center'] - OldCenter))**2))/self.parameters.loc[i, 'NumObservations'])
        # Compute g
        g = np.array(())
        for ni in range(self.parameters.loc[i, 'Dictionary'].shape[0]):
            g = np.append(g, [self.Kernel_Gaussiano(self.parameters.loc[i, 'Dictionary'][ni,],x)])
        G = g.reshape(g.shape[0],1)
        # Computing z
        z = np.matmul(self.parameters.loc[i, 'Q'], g)
        Z = z.reshape(z.shape[0],1)
        # Computing r
        r = self.hyperparameters.loc[0, 'lambda1'] + 1 - np.matmul(Z.T, g)
        # Estimating the error
        EstimatedError = y - np.matmul(G.T, self.parameters.loc[i, 'Theta'])
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters.loc[i, 'Dictionary'].shape[0]):
            distance.append(np.linalg.norm(self.parameters.loc[i, 'Dictionary'][ni,] - x))
        # Finding the index of minimum distance
        IndexMinDistance = np.argmin(distance)
        # Novelty criterion
        if distance[IndexMinDistance] > 0.1 * self.parameters.loc[i, 'nu']:
            self.parameters.at[i, 'Dictionary'] = np.vstack([self.parameters.loc[i, 'Dictionary'], x])
            # Updating Q                      
            self.parameters.at[i, 'Q'] = (1/r)*(self.parameters.loc[i, 'Q']*r + np.matmul(Z,Z.T))
            self.parameters.at[i, 'Q'] = np.lib.pad(self.parameters.loc[i, 'Q'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeQ = self.parameters.loc[i, 'Q'].shape[0] - 1
            self.parameters.at[i, 'Q'][sizeQ,sizeQ] = (1/r)*self.hyperparameters.loc[0, 'omega']
            self.parameters.at[i, 'Q'][0:sizeQ,sizeQ] = (1/r)*(-z)
            self.parameters.at[i, 'Q'][sizeQ,0:sizeQ] = (1/r)*(-z)
            # Updating P
            self.parameters.at[i, 'P'] = np.lib.pad(self.parameters.loc[i, 'P'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i, 'P'].shape[0] - 1
            self.parameters.at[i, 'P'][sizeP,sizeP] = self.hyperparameters.loc[0, 'omega']
            # Updating Theta
            self.parameters.at[i, 'Theta'] = self.parameters.loc[i, 'Theta'] - (Z*(1/r)*EstimatedError)
            self.parameters.at[i, 'Theta'] = np.vstack([self.parameters.loc[i, 'Theta'],(1/r)*EstimatedError])
        else:
            # Calculating q
            q = np.matmul(self.parameters.loc[i, 'P'], Z)/(1 + np.matmul(np.matmul(Z.T, self.parameters.loc[i, 'P']), Z))
            # Updating P
            self.parameters.at[i, 'P'] = self.parameters.loc[i, 'P'] - (np.matmul(np.matmul(np.matmul(self.parameters.loc[i, 'P'],Z), Z.T), self.parameters.loc[i, 'P']))/(1 + np.matmul(np.matmul(Z.T, self.parameters.loc[i, 'P']), Z))
            # Updating Theta
            self.parameters.at[i, 'Theta'] = self.parameters.loc[i, 'Theta'] + np.matmul(self.parameters.loc[i, 'Q'], q) * EstimatedError
            
            
    def Updating_Lambda(self, x):
        # Computing lambda
        TauRules = []
        for i in self.parameters.index:
            Tau = 1
            for j in range(x.shape[0]):
                Tau = Tau * self.Kernel_Gaussiano(x[j], self.parameters.loc[i, 'Center'][0,j])
            TauRules.append(Tau)

        for i,cont in zip(self.parameters.index,range(len(TauRules))):
            self.parameters.at[i, 'SumLambda'] = self.parameters.loc[i, 'SumLambda'] + TauRules[cont]/sum(TauRules)
            
            
    def Utility_Measure(self, x, k):
        # Calculating the utility
        remove = []
        for i in self.parameters.index:
            if (k - self.parameters.loc[i, 'TimeCreation']) == 0:
                self.parameters.at[i, 'Utility'] = 1
            else:
                self.parameters.at[i, 'Utility'] = self.parameters.loc[i, 'SumLambda'] / (k - self.parameters.loc[i, 'TimeCreation'])
            if self.parameters.loc[i, 'Utility'] < self.hyperparameters.loc[0, 'e_utility']:
                remove.append(i)
        if len(remove) > 0:    
            self.parameters = self.parameters.drop(remove)
            # Stoping to creating new rules when the model exclude the first rule
            self.ExcludedRule = 1