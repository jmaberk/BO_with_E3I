# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:51:41 2016

@author: Vu
"""
from __future__ import division
import itertools
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics.pairwise import euclidean_distances

from scipy.optimize import fmin_cobyla

import random
import time

#from ..util.general import multigrid, samples_multidimensional_uniform, reshape

__author__ = 'Vu'

def acq_max_nlopt(ac,gp,y_max,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'NLOPT' library.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    
    try:
        import nlopt
    except:
        print("Cannot find nlopt library")
    
    def objective(x, grad):
            """Objective function in the form required by nlopt."""
            #print "=================================="
            if grad.size > 0:
                fx, gx = ac(x[None], grad=True)
                grad[:] = gx[0][:]
            else:
                try:
                    if ac['name']=='pes':
                        fx = ac(x)
                    else:
                        fx = ac(x,gp,y_max)
                    if isinstance(fx,list):
                        fx=fx[0]
                    
                    #print fx
                except:
                    return 0
            return fx[0]
            
    tol=1e-6
    bounds = np.array(bounds, ndmin=2)

    dim=bounds.shape[0]
    opt = nlopt.opt(nlopt.GN_DIRECT, dim)
    #opt = nlopt.opt(nlopt.LD_MMA, bounds.shape[0])

    opt.set_lower_bounds(bounds[:, 0])
    opt.set_upper_bounds(bounds[:, 1])
    #opt.set_ftol_rel(tol)
    opt.set_maxeval(5000*dim)
    opt.set_xtol_abs(tol)

    #opt.set_ftol_abs(tol)#Set relative tolerance on function value.
    #opt.set_xtol_rel(tol)#Set absolute tolerance on function value.
    #opt.set_xtol_abs(tol) #Set relative tolerance on optimization parameters.

    opt.set_maxtime=5000*dim
    
    opt.set_max_objective(objective)    

    xinit=random.uniform(bounds[:,0],bounds[:,1])
    #xinit=np.asarray(0.2)
    #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
    #print xoptimal
    
    try:
        xoptimal = opt.optimize(xinit.copy())

    except:
        xoptimal=xinit
        #xoptimal = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*1.0 / 2
     
    fmax= opt.last_optimum_value()
    
    #print "nlopt force stop ={:s}".format(nlopt_result)
    #fmax=opt.last_optimize_result()
    
    code=opt.last_optimize_result()
    status=1

    if code<4:
        print "nlopt code = {:d}".format(code)
        status=0

    return xoptimal, fmax, status

    
def acq_max_direct(ac,gp,y_max,bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'DIRECT' library.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    
    try:
        from DIRECT import solve
    except:
        print("Cannot find DIRECT library")
        
    def DIRECT_f_wrapper(ac):
        def g(x, user_data):
            fx=ac(np.array([x]),gp,y_max)
            #print fx[0]
            return fx[0], 0
        return g
            
    lB = np.asarray(bounds)[:,0]
    uB = np.asarray(bounds)[:,1]
    
    #x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=750, maxf=2000,volper=0.005) # this can be used to speed up DIRECT (losses precission)
    x,_,_ = solve(DIRECT_f_wrapper(ac),lB,uB,maxT=2000,maxf=2000,volper=0.0005)
    return np.reshape(x,len(bounds))
    
def acq_max(ac, gp, y_max, bounds, opt_toolbox='scipy',seeds=[]):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """
    
    if opt_toolbox=='nlopt':
        x_max,f_max,status = acq_max_nlopt(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
        
        if status==0:# if nlopt fails, let try scipy
            opt_toolbox='scipy'
            
    if opt_toolbox=='direct':
        x_max = acq_max_direct(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    if opt_toolbox=='scipy':
        x_max = acq_max_scipy(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    if opt_toolbox=='thompson': # thompson sampling
        x_max = acq_max_thompson(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    if opt_toolbox=='cobyla':
        x_max = acq_max_cobyla(ac=ac,gp=gp,y_max=y_max,bounds=bounds)
    if opt_toolbox=='local_search':
        x_max = acq_max_local_search(ac=ac,gp=gp,y_max=y_max,bounds=bounds,seeds=seeds)
    return x_max
        
def acq_max_geometric(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    #myopts ={'maxiter':50}

    def deriv_geometric(x):
        X=gp.X
        #if len(xTest.shape)==1: # 1d
            #xTest=xTest.reshape((-1,X.shape[1]))
            
        Euc_dist=euclidean_distances(x,X)
        #Euc_dist=cdist(xTest,X)
        #Euc_dist=dist_eucledian(xTest,X)

          
        dist=Euc_dist.min(axis=1)
        der=x*dist
        return der


    #myopts ={'maxiter':5*dim,'maxfun':10*dim}
    myopts ={'maxiter':1*dim,'maxfun':2*dim}
    #myopts ={'maxiter':5*dim}

    max_acq = None
    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in xrange(2*dim):
        
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
    
        # evaluate
        #start_eval=time.time()
        y_tries=ac(x_tries,gp=gp, y_max=y_max)
        #end_eval=time.time()
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        x_init_max=x_tries[idx_max]
        
        #start_opt=time.time()
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),bounds=bounds,method="L-BFGS-B",options=myopts)#L-BFGS-B
        #res = fmin_l_bfgs_b(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max), x0=x_init_max.reshape(1, -1), bounds=bounds,fprime =deriv_geometric)
        #res = fmin_l_bfgs_b(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max), x0=x_init_max.reshape(1, -1), 
                            #bounds=bounds,approx_grad=True,options=myopts)
    
        
        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res[0],gp,y_max)        
        else:
            val=ac(res.x,gp,y_max)        
        
        #end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
    
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res[0]
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max


def acq_max_scipy(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':5*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    for i in xrange(2*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
    
        # evaluate
        start_eval=time.time()
        y_tries=ac(x_tries,gp=gp, y_max=y_max)
        end_eval=time.time()
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
        start_opt=time.time()
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp,y_max)        
        else:
            val=ac(res.x,gp,y_max) 

        
        end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max

def acq_max_thompson(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    #myopts ={'maxiter':3*dim,'maxfun':5*dim}
    myopts ={'maxiter':3*dim,'maxfun':5*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in xrange(1*dim):
        # Find the minimum of minus the acquisition function        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(60*dim, dim))
    
        # evaluate
        y_tries=ac(x_tries,gp=gp, y_max=y_max)
        #print "elapse evaluate={:.5f}".format(end_eval-start_eval)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp,y_max)        
        else:
            val=ac(res.x,gp,y_max) 

        
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max
    
def acq_max_with_init(ac, gp, y_max, bounds, init_location=[]):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    dim=bounds.shape[0]
    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    myopts ={'maxiter':5*dim,'maxfun':10*dim}
    #myopts ={'maxiter':5*dim}


    # multi start
    #for i in xrange(5*dim):
    #for i in xrange(1*dim):
    for i in xrange(2*dim):
        # Find the minimum of minus the acquisition function 
        
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(20*dim, dim))
        
        if init_location!=[]:
            x_tries=np.vstack((x_tries,init_location))
        
            
        y_tries=ac(x_tries,gp=gp, y_max=y_max)
        
        #find x optimal for init
        idx_max=np.argmax(y_tries)
        #print "max y_tries {:.5f} y_max={:.3f}".format(np.max(y_tries),y_max)

        x_init_max=x_tries[idx_max]
        
        start_opt=time.time()
    
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),bounds=bounds,
                       method="L-BFGS-B",options=myopts)#L-BFGS-B


        #res = fmin_bfgs(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),x_init_max.reshape(1, -1),disp=False)#L-BFGS-B
        # value at the estimated point
        #val=ac(res.x,gp,y_max)        
        
        if 'x' not in res:
            val=ac(res,gp,y_max)        
        else:
            val=ac(res.x,gp,y_max) 

        
        end_opt=time.time()
        #print "elapse optimize={:.5f}".format(end_opt-start_opt)
        
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val >= max_acq:
            if 'x' not in res:
                x_max = res
            else:
                x_max = res.x
            max_acq = val
            #print max_acq

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    #return np.clip(x_max[0], bounds[:, 0], bounds[:, 1])
        #print max_acq
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def acq_max_local_search(ac, gp, y_max, bounds,seeds):
    """
    A function to find the maximum of the acquisition function using
    the scipy python

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    dim=bounds.shape[0]

    x_max = bounds[:, 0]
    max_acq = None

    #x_tries = np.array([ np.linspace(i,j,500) for i,j in zip( bounds[:, 0], bounds[:, 1])])
    #x_tries=x_tries.T

    #myopts ={'maxiter':2000,'fatol':0.01,'xatol':0.01}
    #myopts ={'maxiter':100,'maxfun':10*dim}
    myopts ={'maxiter':5*dim}

    myidx=np.random.permutation(len(seeds))
    # multi start
    for idx in xrange(5*dim):
    #for i in xrange(1*dim):
    #for idx,xt in enumerate(seeds): 
        xt=seeds[myidx[idx]]
        val=ac(xt,gp,y_max) 
        # Store it if better than previous minimum(maximum).
        if max_acq is None or val > max_acq:
            x_max=xt
            max_acq = val
        #for i in xrange(1*dim):
        for i in xrange(1):
            res = minimize(lambda x: -ac(x, gp=gp, y_max=y_max),xt,bounds=bounds,
                           method="L-BFGS-B",options=myopts)#L-BFGS-B
    
            xmax_temp=np.clip(res.x, bounds[:, 0], bounds[:, 1])
            val=ac(xmax_temp,gp,y_max) 

            # Store it if better than previous minimum(maximum).
            if max_acq is None or val > max_acq:
                x_max = xmax_temp
                max_acq = val
                #print max_acq

    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
    # COBYLA -> x_max[0]
    # L-BFGS-B -> x_max
    

def acq_max_single_seed(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'L-BFGS-B' method.

    Input Parameters
    ----------
    ac: The acquisition function object that return its point-wise value.
    gp: A gaussian process fitted to the relevant data.
    y_max: The current maximum known value of the target function.
    bounds: The variables bounds to limit the search of the acq max.
    
    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    #max_acq = None
    dim=bounds.shape[0]

    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(50*dim, dim))
    
    # evaluate
    y_tries=ac(x_tries,gp=gp, y_max=y_max)
        
    #find x optimal for init
    idx_max=np.argmax(y_tries)
    x_init_max=x_tries[idx_max]
    #x_try=np.array(bounds[:, 0])

    # Find the minimum of minus the acquisition function
    res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                   x_init_max.reshape(1, -1),
                   bounds=bounds,
                   method="L-BFGS-B")

    x_max = res.x
    #max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    
