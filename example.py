'''
Name: example.py
Authors: Julian Berk and Vu Nguyen
Publication date:16/04/2018
Inputs:None
Outputs: Pickle files and plots containing the results from experiments run
Description: Will run a quick example of E3I on the two peak Gaussian mixture 
shown in our paper. It will also print the 1D version of the function for
illustrative purposes
'''
###############################################################################
import sys
sys.path.insert(0,'../../')
from prada_bayes_opt import PradaBayOptFn
import numpy as np
from prada_bayes_opt import auxiliary_functions
#from my_plot_gp import run_experiment
from prada_bayes_opt import functions
from prada_bayes_opt.utility import export_results
import plot_results
from print_two_peak_gaussian import plot_two_peak_gaussian
import pickle
import random
import time
#import pickle
import warnings
import itertools
warnings.filterwarnings("ignore")

'''
***********************************IMPORTANT***********************************
The pickle_location variable below must be changed to the appropriate directory
in your system for the code to work.
'''
#pickle_location='..\..\..'
pickle_location="D:\OneDrive\Documents\PhD\Code\Bayesian\BO_with_E3I\pickleStorage"
###############################################################################

acq_type_list=[]

temp={}
temp['name']='ei'
acq_type_list.append(temp)

temp={}
temp['name']='e3i'
acq_type_list.append(temp)

temp={}
temp['name']='ucb'
acq_type_list.append(temp)

temp={}
temp['name']='ei_zeta'
temp['zeta']=0.01
acq_type_list.append(temp)

mybatch_type_list={'Single'}
plot_two_peak_gaussian()
###############################################################################
'''
#1 The dimension of the two peak Gaussian mixture that will be optimized
#2 num_initial_points controls the number of random sampled points each 
experiment will start with.
#3 max_iterations controls the number of iterations of Bayesian optimization
that will run on the function. This must be controlled with iteration_factor
for compatability with the print function.
#4 num_repeats controls the number of repeat experiments. Higher dimension 
functions can be adjusted separately to avoid long runtimes.
5# acq_params['optimize_gp'] If this is 1, then the lengthscale will be
determined by maximum likelihood every 15 samples. If any other value, no
lengthscale adjustement will be made
'''
###############################################################################

D=2 #1
myfunction_list=[]
myfunction_list.append(functions.doubleGaussian(dim=D))

seed=1
print("Seed of {} used".format(seed))

for idx, (myfunction,acq_type,mybatch_type,) in enumerate(itertools.product(myfunction_list,acq_type_list,mybatch_type_list)):
    func=myfunction.func
    mybound=myfunction.bounds

    yoptimal=myfunction.fmin*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim 
    
    num_initial_points=myfunction.input_dim+1 #2
    
    iteration_factor=20 #3
    max_iterations=iteration_factor*myfunction.input_dim 
    
    if myfunction.input_dim>=5: #4
        num_repeats=10
    else:
        num_repeats=10

    GAP=[0]*num_repeats
    ybest=[0]*num_repeats
    Regret=[0]*num_repeats
    MyTime=[0]*num_repeats
    MyOptTime=[0]*num_repeats
    ystars=[0]*num_repeats

    func_params={}
    func_params['bounds']=myfunction.bounds
    func_params['f']=func

    acq_params={}
    acq_params['acq_func']=acq_type
    acq_params['optimize_gp']=0 #5if 1 then maximum likelihood lenghscale selection will be used 

    for ii in range(num_repeats):
        
        gp_params = {'theta':0.05,'noise_delta':0.001} # Kernel parameters for the square exponential kernel
        baysOpt=PradaBayOptFn(gp_params,func_params,acq_params,experiment_num=ii,seed=seed)

        ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(baysOpt,gp_params,
                                                yoptimal,n_init=num_initial_points,NN=max_iterations)
                                                      
        MyOptTime[ii]=baysOpt.time_opt
        ystars[ii]=baysOpt.ystars
        
    Score={}
    Score["GAP"]=GAP
    Score["ybest"]=ybest
    Score["ystars"]=ystars
    Score["Regret"]=Regret
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    export_results.print_result_ystars(baysOpt,myfunction,Score,mybatch_type,acq_type,toolbox='PradaBO')

#Plots the results. Comment out to supress plots.
for idx, (myfunction) in enumerate(itertools.product(myfunction_list)):
    plot_results.plot(myfunction[0].name,myfunction[0].input_dim,iteration_factor,pickle_location)    