from __future__ import division
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from acquisition_maximization import acq_max
#from spearmint.acquisition_functions import predictive_entropy_search
#from prada_gaussian_process import PradaGaussianProcess


counter = 0


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.acq=acq
        acq_name=acq['name']
        
        ListAcq=['ucb', 'ei','poi','random','ei_mu','e3i','thompson']
        
        # check valid acquisition function
        IsTrue=[val for idx,val in enumerate(ListAcq) if val in acq_name]
        #if  not in acq_name:
        if  IsTrue == []:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(acq_name)
            raise NotImplementedError(err)
        else:
            self.acq_name = acq_name
            
        self.dim=acq['dim']
        
        if 'scalebounds' not in acq:
            self.scalebounds=[0,1]*self.dim
            
        else:
            self.scalebounds=acq['scalebounds']
        
        # vector theta for thompson sampling
        #self.flagTheta_TS=0
        self.initialized_flag=0
        self.objects=[]
        if 'xstars' not in acq:
            self.acq['xstars']=[]
        if 'ystars' not in acq:
            self.acq['ystars']=[]

    def acq_kind(self, x, gp, y_max):

        #print self.kind
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ucb':
            #return self._ucb(x, gp, self.acq['kappa'])
            return self._ucb(x, gp)
        if self.acq_name == 'ei':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'ei_zeta':
            zeta=self.acq['zeta']
            return self._ei_zeta(x, gp, y_max, zeta)
        if self.acq_name == 'e3i':
            if self.initialized_flag==0:# without y* samples
                self.object=AcquisitionFunction.ExplorationEnhancedExpectedImprovement(gp,ystars=self.acq['ystars'])
                self.initialized_flag=1
                return self.object(x)
            else:
                return self.object(x)
        if self.acq_name == 'poi':
            return self._poi(x, gp, y_max)
            #return AcquisitionFunction.ProbabilityOfImprovement(x,gp,y_max)
        if self.acq_name == 'thompson':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.ThompsonSampling(gp)
                self.initialized_flag=1
                return self.object(x,gp)
            else:
                return self.object(x,gp)
       
        if self.acq_name == 'ei_multiple':
            return self._ei_multiple(x, gp, y_max)
        if self.acq_name == 'pure_exploration':
            return self._pure_exploration(x, gp) 
        if self.acq_name == 'pure_exploration_topk':
            return self._pure_exploration_topk(x, gp,self.acq['k']) 
        if self.acq_name == 'ei_mu':
            return self._ei(x, gp, y_max)
        if self.acq_name == 'mu':
            return self._mu(x, gp)
        if self.acq_name == 'geometric':
            return self._geometric(x, gp)
        if self.acq_name == 'ucb_pe':
            return self._ucb_pe(x, gp,self.acq['kappa'],self.acq['maxlcb'])
        if self.acq_name == 'ucb_pe_incremental':
            return self._ucb_pe_incremental(x, gp,self.acq['kappa'],self.acq['maxlcb'])
        if self.acq_name == 'mu*sigma':
            return self._mu_sigma(x, gp)
       
        if self.acq_name == 'pes':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.PredictiveEntropySearch(gp,self.scalebounds,
                                                                             xstars=self.acq['xstars'])
                self.initialized_flag=1
                return self.object(x)
            else:
                return self.object(x)
            
        if self.acq_name == 'es':
            if self.initialized_flag==0:
                self.object=AcquisitionFunction.EntropySearch(gp,self.scalebounds,
                                                              xstars=self.acq['xstars'])
                self.initialized_flag=1
                return self.object(x)
            else:
                return self.object(x)
    def utility_plot(self, x, gp, y_max):
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ei':
            return self._ei_plot(x, gp, y_max)
  
 

    @staticmethod
    def _ucb(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T                
        #return mean + kappa * np.sqrt(var)
        return mean + np.log(len(gp.Y)) * np.sqrt(var)
    @staticmethod
    def _ucb_pe(x, gp, kappa, maxlcb):
        mean, var = gp.predict_bucb(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T

        value=mean + kappa * np.sqrt(var)        
        myidx=[idx for idx,val in enumerate(value) if val<maxlcb]
        var[myidx]=0        
        return var
    
    @staticmethod
    def _ucb_pe_incremental(x, gp, kappa, maxlcb):
        mean, var = gp.predict_bucb_incremental(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T

        value=mean + kappa * np.sqrt(var)
        
        myidx=[idx for idx,val in enumerate(value) if val<maxlcb]
        var[myidx]=0        
        return var
                
    @staticmethod
    def _ei(x, gp, y_max):
        y_max=np.asscalar(y_max)
        mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-8 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-8]=0
        return out
    
    @staticmethod
    def _ei_zeta(x, gp, y_max, zeta):
        y_max=np.asscalar(y_max)+zeta
        mean, var = gp.predict(x, eval_MSE=True)
        var2 = np.maximum(var, 1e-8 + 0 * var)
        z = (mean - y_max)/np.sqrt(var2)        
        out=(mean - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)
        
        out[var2<1e-8]=0
        return out
    
    @staticmethod
    def _poi(x, gp,y_max): # run Predictive Entropy Search using Spearmint
        mean, var = gp.predict(x, eval_MSE=True)    
        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)
        z = (mean - y_max)/np.sqrt(var)        
        return norm.cdf(z)  

    @staticmethod
    def _mu_sigma(x, gp):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean*var
    class ExplorationEnhancedExpectedImprovement(object):
        """
        Calculates the E3I acquisition function values
        Inputs: gp: The Gaussian process, also contains all data
                ystars: The pre-calculated Thompson sample maxima
                x:The point at which to evaluate the acquisition function 
        Output: acq_value: The value of the aquisition function at point x
        """
        def __init__(self,gp,ystars=[]):
            self.X=gp.X
            self.Y=gp.Y
            self.gp=gp
            if ystars==[]:
                print "y_star is empty for MES"                
            self.y_stars=ystars
                
        def __call__(self,x):
            mean_x, var_x = self.gp.predict(x, eval_MSE=True)

            var2 = np.maximum(var_x, 1e-8)

            acq_value=np.asarray([0]*len(var_x))
        
            for idx,y_max in enumerate(self.y_stars):
                z = (mean_x - y_max)/np.sqrt(var2)        
                out=(mean_x - y_max) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z)

                acq_value=acq_value+out
                
            acq_value[var2<1e-8]=0

            return acq_value
    class ThompsonSampling(object):
        """
        Class used for calulating Thompson samples. Re-usable calculations are
        done in __init__ to reduce compuational cost.
        """
        #Calculates the thompson sample paramers 
        def __init__(self,gp):
            dim=gp.X.shape[1]
            # used for Thompson Sampling
            self.WW_dim=200 # dimension of random feature
            self.WW=np.random.multivariate_normal([0]*self.WW_dim,np.eye(self.WW_dim),dim)/gp.lengthscale  
            self.bias=np.random.uniform(0,2*3.14,self.WW_dim)

            # computing Phi(X)^T=[phi(x_1)....phi(x_n)]
            Phi_X=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(gp.X,self.WW)+self.bias), np.cos(np.dot(gp.X,self.WW)+self.bias)]) # [N x M]
            
            # computing A^-1
            A=np.dot(Phi_X.T,Phi_X)+np.eye(2*self.WW_dim)*gp.noise_delta
            gx=np.dot(Phi_X.T,gp.Y)
            self.mean_theta_TS=np.linalg.solve(A,gx)
        #Calculates the thompson sample value at the point x    
        def __call__(self,x,gp):
            #phi_x=np.sqrt(1.0/self.UU_dim)*np.hstack([np.sin(np.dot(x,self.UU)), np.cos(np.dot(x,self.UU))])
            phi_x=np.sqrt(2.0/self.WW_dim)*np.hstack([np.sin(np.dot(x,self.WW)+self.bias), np.cos(np.dot(x,self.WW)+self.bias)])
            
            # compute the mean of TS
            gx=np.dot(phi_x,self.mean_theta_TS)    
            return gx    
    # for plot purpose
    @staticmethod
    def _ei_plot(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)        
        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)           
            z = (mean - y_max)/np.sqrt(var)
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)           
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP           
            prod_out=[1]*len(mean[0])
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])    
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                prod_out=prod_out*out[idx]
            out=np.asarray(out)           
            #return np.mean(out,axis=0) # mean over acquisition functions
            return np.prod(out,axis=0) # product over acquisition functions
        
  
    
def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]



class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'
