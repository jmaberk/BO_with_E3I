'''
Name: bayesianOptimizationMaster.py
Authors: Julian Berk and Vu Nguyen
Publication date:16/04/2018
Inputs:None
Outputs: Pickle files and plots containing the results from experiments run
Description: The master file for code used to generate the results for the
paper Exploration Enhanced Expected Improvement for Bayesian Optimization.
Most aspects of the algorithm can be altered from this file. See comments for
more details
'''
###############################################################################
import sys
sys.path.insert(0,'../../')
from prada_bayes_opt import PradaBayOptFn
import numpy as np
from prada_bayes_opt import auxiliary_functions
#from my_plot_gp import run_experiment
from prada_bayes_opt import functions
from prada_bayes_opt import real_experiment_function
from prada_bayes_opt.utility import export_results
import plot_results
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
'''
Here the user can choose which functions to optimize. Just un-comment the 
desired functions and set the desired dimensions with the dim parameter
in supported functions
'''
###############################################################################
myfunction_list=[]
myfunction_list.append(functions.doubleGaussian(dim=1))
#myfunction_list.append(functions.gaussian(dim=8))
#myfunction_list.append(functions.mixture(peaks=3))
#myfunction_list.append(functions.beale())
#myfunction_list.append(functions.forrester())
#myfunction_list.append(functions.rosenbrock())
#myfunction_list.append(functions.eggholder())
#myfunction_list.append(functions.franke())
#myfunction_list.append(functions.shubert())
#myfunction_list.append(functions.schwefel(dim=4))
#myfunction_list.append(functions.griewank(dim=3))
#myfunction_list.append(functions.levy(dim=5))
#myfunction_list.append(functions.branin())
#myfunction_list.append(functions.dropwave())
#myfunction_list.append(functions.sixhumpcamel())
#myfunction_list.append(functions.hartman_3d())
#myfunction_list.append(functions.ackley(input_dim=5))
#myfunction_list.append(functions.alpine1(input_dim=5))
#myfunction_list.append(functions.alpine2(input_dim=5))
#myfunction_list.append(functions.hartman_6d())
#myfunction_list.append(functions.alpine2(input_dim=10))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1])))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1,1,1])))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])))
#myfunction_list.append(real_experiment_function.SVR_function())
#myfunction_list.append(real_experiment_function.AlloyCooking_Profiling_3Steps())
#myfunction_list.append(real_experiment_function.Robot_BipedWalker())
#myfunction_list.append(real_experiment_function.DeepLearning_MLP_MNIST())
#myfunction_list.append(real_experiment_function.BayesNonMultilabelClassification())
#myfunction_list.append(real_experiment_function.BayesNonMultilabelClassificationEnron())

    
###############################################################################
'''
Here the user can choose which acquisition functions will be used. To select
an acquisition function, un-comment the "acq_type_list.append(temp)" after its
name. If you do not have any pickle files for the method and function, you will
also need to comment out the relevent section in plot_results.py.
'''
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
temp['kappa']=2
acq_type_list.append(temp)

temp={}
temp['name']='ei_zeta'
temp['zeta']=0.01
acq_type_list.append(temp)

mybatch_type_list={'Single'}

###############################################################################
'''
#1 seed is used along with the experiment number as a seed to randomly generate
the initial points. Setting this as a constant will allow results to be
reproduced while making it random will let each set of runs use a different
set of initial points.
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

#seed=np.random.randint(1,100) #1
seed=1
print("Seed of {} used".format(seed))

for idx, (myfunction,acq_type,mybatch_type,) in enumerate(itertools.product(myfunction_list,acq_type_list,mybatch_type_list)):
    func=myfunction.func
    mybound=myfunction.bounds
    #gp_params = {'theta':0.05,'noise_delta':0.001}
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
        
    #baysOpt.xt_suggestions=[]
    #baysOpt.y
    Score={}
    Score["GAP"]=GAP
    Score["ybest"]=ybest
    Score["ystars"]=ystars
    Score["Regret"]=Regret
    Score["MyTime"]=MyTime
    Score["MyOptTime"]=MyOptTime
    export_results.print_result_ystars(baysOpt,myfunction,Score,mybatch_type,acq_type,toolbox='PradaBO')
for idx, (myfunction) in enumerate(itertools.product(myfunction_list)):
    plot_results.plot(myfunction[0].name,myfunction[0].input_dim,iteration_factor,pickle_location)    