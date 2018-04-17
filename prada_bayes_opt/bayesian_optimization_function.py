# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""

from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
#from visualization import Visualization
from prada_gaussian_process import PradaGaussianProcess
#from prada_gaussian_process import PradaMultipleGaussianProcess

from acquisition_maximization import acq_max_nlopt
from acquisition_maximization import acq_max_direct
from acquisition_maximization import acq_max
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import pickle
import time
import copy
import math
import random
#import nlopt

#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
counter = 0


class PradaBayOptFn(object):

    def __init__(self, gp_params, func_params, acq_params, experiment_num, seed, verbose=1):
        """      
        Input parameters
        ----------
        
        gp_params:                  GP parameters
        gp_params.theta:            to compute the kernel
        gp_params.delta:            to compute the kernel
        
        func_params:                function to optimize
        func_params.init bound:     initial bounds for parameters
        func_params.bounds:        bounds on parameters        
        func_params.func:           a function to be optimized
        
        
        acq_params:            acquisition function, 
        acq_params.acq_func['name']=['ei','ucb','poi','lei']
                            ,acq['kappa'] for ucb, acq['k'] for lei
        acq_params.opt_toolbox:     optimization toolbox 'nlopt','direct','scipy'
        
        experiment_num: the interation of the GP method. Used to make sure each 
                        independant stage of the experiment uses different 
                        initial conditions
        seed: Variable used as part of a seed to generate random initial points
                            
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """

        self.experiment_num=experiment_num
        self.seed=seed
        
        # Find number of parameters
        bounds=func_params['bounds']
        if 'init_bounds' not in func_params:
            init_bounds=bounds
        else:
            init_bounds=func_params['init_bounds']
        
        self.dim = len(bounds)

        # Create an array with parameters bounds
        if isinstance(bounds,dict):
            # Get the name of the parameters
            self.keys = list(bounds.keys())
        
            self.bounds = []
            for key in bounds.keys():
                self.bounds.append(bounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(bounds)

        if len(init_bounds)==0:
            self.init_bounds=self.bounds.copy()
        else:
            self.init_bounds=init_bounds
            
        if isinstance(init_bounds,dict):
            # Get the name of the parameters
            self.keys = list(init_bounds.keys())
        
            self.init_bounds = []
            for key in init_bounds.keys():
                self.init_bounds.append(init_bounds[key])
            self.init_bounds = np.asarray(self.init_bounds)
        else:
            self.init_bounds=np.asarray(init_bounds)            
            
        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        # Some function to be optimized
        self.f = func_params['f']
        # optimization toolbox
        if 'opt_toolbox' not in acq_params:
            self.opt_toolbox='scipy'
        else:
            self.opt_toolbox=acq_params['opt_toolbox']
        # acquisition function type
        
        self.acq=acq_params['acq_func']
        self.acq['scalebounds']=self.scalebounds
        
        if 'debug' not in self.acq:
            self.acq['debug']=0           
        if 'stopping' not in acq_params:
            self.stopping_criteria=0
        else:
            self.stopping_criteria=acq_params['stopping']
        if 'optimize_gp' not in acq_params:
            self.optimize_gp=0
        else:                
            self.optimize_gp=acq_params['optimize_gp']       
        if 'marginalize_gp' not in acq_params:
            self.marginalize_gp=0
        else:                
            self.marginalize_gp=acq_params['marginalize_gp']
            
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        # (y - mean)/(max-min)
        self.Y = None
               
        # y original scale
        self.Y_original = None
        
        # value of the acquisition function at the selected point
        self.alpha_Xt=None
        self.Tau_Xt=None
        
        self.time_opt=0

        self.k_Neighbor=2
        
        # Lipschitz constant
        self.L=0
        
        # Gaussian Process class
        self.gp=PradaGaussianProcess(gp_params)

        # acquisition function
        self.acq_func = None
    
        # stop condition
        self.stop_flag=0
        self.logmarginal=0
        
        # xt_suggestion, caching for Consensus
        self.xstars=[]
        self.ystars=np.zeros((2,1))
        
        # theta vector for marginalization GP
        self.theta_vector =[]
        
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self,gp_params, n_init_points=3):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """
        # set seed to allow for reproducible results
        np.random.seed(self.experiment_num*self.seed)
        print(self.experiment_num)
        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in self.init_bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))
        
        self.X_original = np.asarray(init_X)
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))

        self.Y_original = np.asarray(y_init)        
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)#/(np.max(self.Y_original)-np.min(self.Y_original))

        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X = np.asarray(temp_init_point)
		
        
        
    def maximize(self,gp_params):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
            
        if self.acq['name']=='random':
            x_max = [np.random.uniform(x[0], x[1], size=1) for x in self.bounds]
            x_max=np.asarray(x_max)
            x_max=x_max.T
            self.X_original=np.vstack((self.X_original, x_max))
            # evaluate Y using original X
            
            #self.Y = np.append(self.Y, self.f(temp_X_new_original))
            self.Y_original = np.append(self.Y_original, self.f(x_max))
            
            # update Y after change Y_original
            self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
            
            self.time_opt=np.hstack((self.time_opt,0))
            return         

        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        if self.gp.KK_x_x_inv ==[]:
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

 
        acq=self.acq

        if acq['debug']==1:
            logmarginal=self.gp.log_marginal_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            print(gp_params['theta'])
            print("log marginal before optimizing ={:.4f}".format(logmarginal))
            self.logmarginal=logmarginal
                
            if logmarginal<-999999:
                logmarginal=self.gp.log_marginal_lengthscale(gp_params['theta'],gp_params['noise_delta'])

        
        # optimize GP parameters after 5 iterations
        if self.optimize_gp==1 and len(self.Y)%10*self.dim==0:
            print("Initial length scale={}".format(gp_params['theta']))
            newtheta = self.gp.optimize_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            gp_params['theta']=newtheta
            print("New length scale={}".format(gp_params['theta']))
            #logmarginal=self.gp.log_marginal_lengthscale(newtheta,gp_params['noise_delta'])
            #print "print newtheta={:s} log marginal={:.4f}".format(newtheta,logmarginal)
            
            # init a new Gaussian Process after optimizing hyper-parameter
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
        
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()
        
        if acq['name'] in ['consensus','mes']: 
            ucb_acq_func={}
            ucb_acq_func['name']='ucb'
            ucb_acq_func['kappa']=np.log(len(self.Y))
            ucb_acq_func['dim']=self.dim
            ucb_acq_func['scalebounds']=self.scalebounds
        
            myacq=AcquisitionFunction(ucb_acq_func)
            xt_ucb = acq_max(ac=myacq.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
            
            xstars=[]
            xstars.append(xt_ucb)
            
            ei_acq_func={}
            ei_acq_func['name']='ei'
            ei_acq_func['dim']=self.dim
            ei_acq_func['scalebounds']=self.scalebounds
        
            myacq=AcquisitionFunction(ei_acq_func)
            xt_ei = acq_max(ac=myacq.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
            xstars.append(xt_ei)
                 
            
            pes_acq_func={}
            pes_acq_func['name']='pes'
            pes_acq_func['dim']=self.dim
            pes_acq_func['scalebounds']=self.scalebounds
        
            myacq=AcquisitionFunction(pes_acq_func)
            xt_pes = acq_max(ac=myacq.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
            xstars.append(xt_pes)
            
            
            self.xstars=xstars            
            
        if acq['name']=='vrs':
            print("please call the maximize_vrs function")
            return
                      
        if 'xstars' not in globals():
            xstars=[]
            
        self.xstars=xstars

        self.acq['xstars']=xstars
        self.acq_func = AcquisitionFunction(self.acq)

        if acq['name']=="ei_mu":
            #find the maximum in the predictive mean
            mu_acq={}
            mu_acq['name']='mu'
            mu_acq['dim']=self.dim
            acq_mu=AcquisitionFunction(mu_acq)
            x_mu_max = acq_max(ac=acq_mu.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox)
            # set y_max = mu_max
            y_max=acq_mu.acq_kind(x_mu_max,gp=self.gp, y_max=y_max)

        
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)

        if acq['name']=='consensus' and acq['debug']==1: # plot the x_max and xstars
            fig=plt.figure(figsize=(5, 5))

            plt.scatter(xt_ucb[0],xt_ucb[1],marker='s',color='g',s=200,label='Peak')
            plt.scatter(xt_ei[0],xt_ei[1],marker='s',color='k',s=200,label='Peak')
            plt.scatter(x_max[0],x_max[1],marker='*',color='r',s=300,label='Peak')
            plt.xlim(0,1)
            plt.ylim(0,1)
            strFileName="acquisition_functions_debug.eps"
            fig.savefig(strFileName, bbox_inches='tight')

        if acq['name']=='vrs' and acq['debug']==1: # plot the x_max and xstars
            fig=plt.figure(figsize=(5, 5))

            plt.scatter(xt_ucb[0],xt_ucb[1],marker='s',color='g',s=200,label='Peak')
            plt.scatter(xt_ei[0],xt_ei[1],marker='s',color='k',s=200,label='Peak')
            plt.scatter(x_max[0],x_max[1],marker='*',color='r',s=300,label='Peak')
            plt.xlim(0,1)
            plt.ylim(0,1)
            strFileName="vrs_acquisition_functions_debug.eps"
            #fig.savefig(strFileName, bbox_inches='tight')
            
            
        val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
        #print x_max
        #print val_acq
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)

            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
        
        self.alpha_Xt= np.append(self.alpha_Xt,val_acq)
        
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
        if self.gp.flagIncremental==1:
            self.gp.fit_incremental(x_max,self.Y[-1])

    def maximize_ei_dist(self,gp_params):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
               
        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])

        
        # optimize GP lengthscale after 15 iterations if self.optimize_gp==1
        if self.optimize_gp==1 and len(self.Y)%15==0:
            print("Initial length scale={}".format(gp_params['theta']))
            newtheta = self.gp.optimize_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            gp_params['theta']=newtheta
            print("New length scale={}".format(newtheta))
            
            # init a new Gaussian Process using the new theta
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to prevent GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()                       
        self.xstars=[]
        self.y_stars=[]
        
        y_max=np.max(self.Y)
###############################################################################        
# numXtar controls the number of thompson samples, given as M in the paper
###############################################################################
        numXtar=100
        
        temp=[]
        # finding the xt of Thompson Sampling
        ii=0
        while ii<numXtar:
            mu_acq={}
            mu_acq['name']='thompson'
            mu_acq['dim']=self.dim
            mu_acq['scalebounds']=self.scalebounds     
            acq_mu=AcquisitionFunction(mu_acq)
            #Get the location of the Thompson sample maxima
            xt_TS = acq_max(ac=acq_mu.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox='scipy')
            # get the value g*
            y_xt_TS=acq_mu.acq_kind(xt_TS,self.gp,y_max=y_max)
            temp.append(xt_TS)
            self.xstars.append(xt_TS)
            self.y_stars.append(y_xt_TS)
            ii+=1                                

        if self.acq['debug']==1:
            print('mean y*={:.4f}({:.8f}) y+={:.4f}'.format(np.mean(y_xt_TS),np.std(y_xt_TS),y_max))
            
        if self.xstars==[]:
            self.xstars=temp
        #save optimal Thompson sample mean and sdv for later analysis
        y_stars=np.array([np.mean(self.y_stars),np.std(self.y_stars)]).reshape(2,-1)
        self.acq['xstars']=self.xstars   
        self.acq['ystars']=self.y_stars   
        self.ystars=np.hstack((self.ystars,(np.array(y_stars))))
        self.acq_func = AcquisitionFunction(self.acq)
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.ystars)
        val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
        
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
        
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)
        
    def maximize_vrs_of_ts(self,gp_params):
        """
        Main optimization method.

        Input parameters
        ----------
        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        if self.stop_flag==1:
            return
               
        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])

        acq=self.acq

        if acq['debug']==1:
            logmarginal=self.gp.log_marginal_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            print(gp_params['theta'])
            print("log marginal before optimizing ={:.4f}".format(logmarginal))
            self.logmarginal=logmarginal
                
            if logmarginal<-999999:
                logmarginal=self.gp.log_marginal_lengthscale(gp_params['theta'],gp_params['noise_delta'])
        
        # optimize GP parameters after 5 iterations
        #if self.optimize_gp==1 and len(self.Y)>=6*self.dim and len(self.Y)%7*self.dim==0:
        if self.optimize_gp==1 and len(self.Y)%(4*self.dim)==0:
            newtheta = self.gp.optimize_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            gp_params['theta']=newtheta    
            logmarginal=self.gp.log_marginal_lengthscale(newtheta,gp_params['noise_delta'])
            if acq['debug']==1:
                print("{:s} log marginal={:.4f}".format(newtheta,logmarginal))
                
        if 'n_xstars' in self.acq:
            numXstar=self.acq['n_xstars']
        else:
            numXstar=10*self.dim
            
        if self.marginalize_gp==1 and len(self.Y)==(5*self.dim):
            newtheta = self.gp.optimize_lengthscale(gp_params['theta'],gp_params['noise_delta'])
            gp_params['theta']=newtheta  
 
        if self.marginalize_gp==1 and len(self.Y)%(8*self.dim)==0:
            self.theta_vector = self.gp.slice_sampling_lengthscale_SE(gp_params['theta'],gp_params['noise_delta'],nSamples=numXstar)
            #gp_params['theta']=newtheta   
            gp_params['newtheta_vector']=self.theta_vector    

            #print newtheta_vector
            #logmarginal=self.gp.log_marginal_lengthscale(newtheta,gp_params['noise_delta'])
            #print "{:s} log marginal={:.4f}".format(newtheta,logmarginal)
        
        # Set acquisition function
        start_opt=time.time()

        y_max = self.Y.max()                
        
        # run the acquisition function for the first time to get xstar
        
        self.xstars=[]
        # finding the xt of UCB
        
        y_max=np.max(self.Y)
        
        temp=[]
        # finding the xt of Thompson Sampling
        for ii in range(numXstar):
            if self.theta_vector!=[]:
                gp_params['theta']=self.theta_vector[ii]    

            # init a new Gaussian Process
            self.gp=PradaGaussianProcess(gp_params)
            # Find unique rows of X to avoid GP from breaking
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])       
        
            mu_acq={}
            mu_acq['name']='thompson'
            mu_acq['dim']=self.dim
            mu_acq['scalebounds']=self.scalebounds     
            acq_mu=AcquisitionFunction(mu_acq)
            xt_TS = acq_max(ac=acq_mu.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox='thompson')
            
            # get the value f*
            y_xt_TS=acq_mu.acq_kind(xt_TS,self.gp,y_max=y_max)
            
            temp.append(xt_TS)
            # check if f* > y^max and ignore xt_TS otherwise
            #if y_xt_TS>=y_max:
                #self.xstars.append(xt_TS)

        if self.xstars==[]:
            #print 'xt_suggestion is empty'
            # again perform TS and take all of them

            self.xstars=temp
        
        
        # check predictive variance before adding a new data points
        var_before=self.gp.compute_var(self.X,self.xstars) 
        var_before=np.mean(var_before)
        
        self.gp.lengthscale_vector=self.theta_vector
        self.acq['xstars']=self.xstars    
        self.acq_func = AcquisitionFunction(self.acq)
        x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox=self.opt_toolbox,seeds=self.xstars)
        #xstars_array=np.asarray(self.acq_func.object.xstars)
    
        val_acq=-self.acq_func.acq_kind(x_max,self.gp,y_max)
        
        
        # check predictive variance after
        temp=np.vstack((self.gp.X,x_max))
        var_after=self.gp.compute_var(temp,self.xstars) 
        var_after=np.mean(var_after)
        print("predictive variance before={:.12f} after={:.12f} val_acq={:.12f}".format(var_before,var_after,np.asscalar(val_acq)))
        
        
        
        # check maximum variance
        var_acq={}
        var_acq['name']='pure_exploration'
        var_acq['dim']=self.dim
        var_acq['scalebounds']=self.scalebounds     
        acq_var=AcquisitionFunction(var_acq)
        temp = acq_max(ac=acq_var.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds,opt_toolbox='scipy')
        
        # get the value f*
        max_var_after=acq_var.acq_kind(temp,self.gp,y_max=y_max)
        print("max predictive variance ={:.8f}".format(np.asscalar(max_var_after)))

        
        if self.stopping_criteria!=0 and val_acq<self.stopping_criteria:
            val_acq=self.acq_func.acq_kind(x_max,self.gp,y_max)
            self.stop_flag=1
            print("Stopping Criteria is violated. Stopping Criteria is {:.15f}".format(self.stopping_criteria))
        
                
        mean,var=self.gp.predict(x_max, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-20]=0
        #self.Tau_Xt= np.append(self.Tau_Xt,val_acq/var)
       
        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        
        #self.Y = np.append(self.Y, self.f(temp_X_new_original))
        self.Y_original = np.append(self.Y_original, self.f(temp_X_new_original))
        
        # update Y after change Y_original
        self.Y=(self.Y_original-np.mean(self.Y_original))/np.std(self.Y_original)

#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================
