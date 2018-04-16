# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 21:37:03 2016

@author: Vu
"""
#from prada_bayes_opt import PradaBayOptFn

#from sklearn.gaussian_process import GaussianProcess
#from scipy.stats import norm
#import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D

from prada_bayes_opt.acquisition_functions import AcquisitionFunction
from prada_bayes_opt import bayesian_optimization_function
import matplotlib.pyplot as plt
from matplotlib import gridspec
#from bayes_opt import PradaBayesianOptimization
import numpy as np
import random
import time
import pickle
import os
import sys
from prada_bayes_opt import PradaBayOptFn
from prada_gaussian_process import PradaGaussianProcess
#from plot_thompson_samples import printThompson
       
def run_experiment(bo,gp_params,yoptimal=0,n_init=3,NN=10):
    """
    Control loop that runs the BO method for the given function

    Input parameters
    ----------
    bo: The Bayesian Optimization object
    gp_params: parameter for Gaussian Process
    yoptimal: Not used
    n_init: Number of initial points
    NN: Number of iterations. T in the paper

    Returns
    -------
    fxoptimal: The y values at each iterations
    elapsed_time: The duration of the experiment
    """
    # create an empty object for BO
    
    start_time = time.time()
    bo.init(gp_params,n_init_points=n_init)
    
    # number of recommended parameters
    for index in range(0,NN-1):
        #print index
        
        if bo.acq['name']=='e3i':
            bo.maximize_ei_dist(gp_params)
        else:
            bo.maximize(gp_params)
    

    fxoptimal=bo.Y_original
    elapsed_time = time.time() - start_time

    return fxoptimal, elapsed_time
            
def yBest_Iteration(YY,BatchSzArray,IsPradaBO=0,Y_optimal=0,step=3):
    """
    Used for finding the

    Input parameters
    ----------
    bo: The Bayesian Optimization object
    gp_params: parameter for Gaussian Process
    yoptimal: Not used
    n_init: Number of initial points
    NN: Number of iterations. T in the paper

    Returns
    -------
    mean_TT: The mean optimal y values at each iterations over all
    experiments
    std_TT:The sdv of the optimal y values at each iterations over all
    experiments
    mean_cum_TT: not used
    std_cum_TT: not used
    """
    nRepeat=len(YY)
    YY=np.asarray(YY)
    YY=np.vstack(YY.T)
    YY=YY.T
    print YY.shape
    ##YY_mean=np.mean(YY,axis=0)
    #YY_std=np.std(YY,axis=0)
    
    mean_TT=[]
    #temp_std=np.std(YY[:,0:BatchSzArray[0]+1])
    #temp_std=np.std(YY_mean[0:BatchSzArray[0]+1])
    
    mean_cum_TT=[]
    
    for idxtt,tt in enumerate(range(0,nRepeat)): # TT run
    
        if IsPradaBO==1:
            temp_mean=YY[idxtt,0:BatchSzArray[0]+1].max()
        else:
            temp_mean=YY[idxtt,0:BatchSzArray[0]+1].min()
        
        temp_mean_cum=YY[idxtt,0:BatchSzArray[0]+1].mean()

        start_point=0
        for idx,bz in enumerate(BatchSzArray): # batch
            if idx==len(BatchSzArray)-1:
                break
            bz=np.int(bz)

            #    get the average in this batch
            temp_mean_cum=np.vstack((temp_mean_cum,YY[idxtt,start_point:start_point+bz].mean()))
            
            # find maximum in each batch            
            if IsPradaBO==1:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].max()))
            else:
                temp_mean=np.vstack((temp_mean,YY[idxtt,start_point:start_point+bz].min()))

            start_point=start_point+bz

        if IsPradaBO==1:
            myYbest=[temp_mean[:idx+1].max()*-1 for idx,val in enumerate(temp_mean)]
            temp_mean_cum=temp_mean_cum*-1
            temp_mean=temp_mean*-1
        else:
            myYbest=[temp_mean[:idx+1].min() for idx,val in enumerate(temp_mean)]
        
        # cumulative regret for each independent run
        #myYbest_cum=[np.mean(np.abs(temp_mean_cum[:idx+1]-Y_optimal)) for idx,val in enumerate(temp_mean_cum)]
        
        temp_regret=np.abs(temp_mean-Y_optimal)
        myYbest_cum=[np.mean(temp_regret[:idx+1]) for idx,val in enumerate(temp_regret)]


        if len(mean_TT)==0:
            mean_TT=myYbest
            mean_cum_TT=myYbest_cum
        else:
            #mean_TT.append(temp_mean)
            mean_TT=np.vstack((mean_TT,myYbest))
            mean_cum_TT=np.vstack((mean_cum_TT,myYbest_cum))
            
    mean_TT    =np.array(mean_TT)
    std_TT=np.std(mean_TT,axis=0)
    std_TT=np.array(std_TT).ravel()
    mean_TT=np.mean(mean_TT,axis=0)

    
    mean_cum_TT=np.array(mean_cum_TT)   
    std_cum_TT=np.std(mean_cum_TT,axis=0)
    std_cum_TT=np.array(std_cum_TT).ravel()
    mean_cum_TT=np.mean(mean_cum_TT,axis=0)
    #return mean_TT[::step],std_TT[::step]#,mean_cum_TT[::5],std_cum_TT[::5]
    return mean_TT[::step],std_TT[::step],mean_cum_TT[::step],std_cum_TT[::step]
    
