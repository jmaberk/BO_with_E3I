# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:35:55 2018

@author: Julian
"""
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
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prada_bayes_opt import visualization
#from prada_bayes_opt import auxiliary_functions
from prada_bayes_opt.utility import export_results
import matplotlib
from prada_bayes_opt import functions
import print_functions
function_name='doubleGaussian'
D=1
#np.random.seed(1)
n_init_points=10
init_bounds=[(0,1)]*D
# Generate random points
l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in init_bounds]
x=np.arange(0,1,0.01).reshape(-1,1)
# Concatenate new random points to possible existing
 #points from self.explore method.
f=functions.doubleGaussian(dim=D)
D=2
y_optimal_value=0
temp=np.asarray(l)
xTrial=temp.reshape(-1,1)
yTrial=f.func(xTrial).T
gp_params = {'theta':0.1,'noise_delta':0.0001}
gp=PradaGaussianProcess(gp_params)
gp.fit(xTrial,yTrial)
TS=AcquisitionFunction.ThompsonSampling(gp)
Y_Ts=TS(x,gp)
plt.plot(x,Y_Ts)
plt.scatter(xTrial,yTrial,color='r')
step=2
T=10*D+1
x_axis=np.array(range(0,T+1))
x_axis=x_axis[::step]
'''
# is minimization problem
IsMin=1
#IsMin=-1
IsLog=0
np.random.seed(1)
n_init_points=10
init_bounds=[(0,1)]*D
# Generate random points
l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in init_bounds]
# Concatenate new random points to possible existing
 #points from self.explore method.
f=functions.doubleGaussian(dim=D)
temp=np.asarray(l)
xTrial=temp=temp.T
yTrial=f.func(xTrial).T
gp_params = {'theta':0.1,'noise_delta':0.0001}
gp=PradaGaussianProcess(gp_params)
gp.fit(xTrial,yTrial)
TS=AcquisitionFunction.ThompsonSampling(gp)

#g=TS(y,gp)
x1=np.arange(0,1,0.01)
x2=np.arange(0,1,0.01)
#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
X1,X2=np.meshgrid(x1,x2)
g=np.array([TS([i,j],gp) for i,j in zip(np.ravel(X1),np.ravel(X2))])
G=g.reshape(X1.shape)
print("TS max: {}".format(np.max(G)))
#ax.plot_surface(X1,X2,Y)
plt.contourf(X1,X2,G)
plt.colorbar()
plt.scatter(xTrial[:,0],xTrial[:,1],color='r')
plt.show()'''
print_functions