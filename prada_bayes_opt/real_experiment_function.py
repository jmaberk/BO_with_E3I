# -*- coding: utf-8 -*-
"""
Name: real_experiment_functions.py
Authors: Julian Berk and Vu Nguyen
Publication date:16/04/2018
Description: These classes run real-world experiments that can be used to test
our acquisition functions

###############################IMPORTANT#######################################
The classes here all have file paths that need to be set correctlt for them to
work. Please make sure you change all paths before using a class
"""

import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVR
import math

import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
        
        
def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class functions:
    def plot(self):
        print "not implemented"
        
    
    
class sincos(functions):
    def __init__(self):
        self.input_dim=1
        self.bounds={'x':(-2,12)}
        self.fmin=11
        self.min=0
        self.ismax=1
        self.name='sincos'
    def func(self,x):
        x=np.asarray(x)

        fval=x*np.sin(x)+x*np.cos(2*x)
        return fval*self.ismax

class fourier(functions):
	'''
	Forrester function. 
	
	:param sd: standard deviation, to generate noisy evaluations of the function.
	'''
	def __init__(self,sd=None):
		self.input_dim = 1		
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = 4.795 		## approx
		self.fmin = -9.5083483926941064 			## approx
		self.bounds = {'x':(0,10)}
		self.name='sincos'
		self.ismax=-1

	def func(self,X):
		X = X.reshape((len(X),1))
		n = X.shape[0]
		fval = X*np.sin(X)+X*np.cos(2*X)
		if self.sd ==0:
			noise = np.zeros(n).reshape(n,1)
		else:
			noise = np.random.normal(0,self.sd,n).reshape(n,1)
		return self.ismax*fval.reshape(n,1) + noise
        
        
class branin(functions):
    def __init__(self):
        self.input_dim=2
        self.bounds=OrderedDict([('x1',(-5,10)),('x2',(-5,10))])
        self.fmin=0.397887
        self.min=[9.424,2.475]
        self.ismax=-1
        self.name='branin'
    #def func(self,x1,x2):
    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        a=1
        b=5.1/(4*np.pi*np.pi)
        c=5/np.pi
        r=6
        s=10
        t=1/(8*np.pi)
        fx=a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s    
        return fx*self.ismax
        
class SVR_function:
    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None: 
            self.bounds = OrderedDict([('C',(0.1,1000)),('epsilon',(0.000001,1)),('gamma',(0.00001,5))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='SVR_function'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_SVR(self,X,X_train,y_train,X_test,y_test):
        x1=X[0]
        x2=X[1]
        x3=X[2]
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        #nTest=X_test.shape[0]
    
        #print x1,x2,x3
        # Fit regression model
        svr_model = SVR(kernel='rbf', C=x1, epsilon=x2,gamma=x3)
        y_pred = svr_model.fit(X_train, y_train).predict(X_test)
        
        
        squared_error=y_pred-y_test
        squared_error=np.mean(squared_error**2)
        
        RMSE=np.sqrt(squared_error)
        return RMSE
        
    def func(self,X):
        X=np.asarray(X)
        ##########################CHANGE PATH##################################    
        Xdata, ydata = self.get_data("D:\\OneDrive\\Documents\\PhD\Code\\Bayesian\\PradaBayesianOptimization\\real_experiment\\space_ga_scale")
        nTrain=np.int(0.7*len(ydata))
        X_train, y_train = Xdata[:nTrain], ydata[:nTrain]
        X_test, y_test = Xdata[nTrain+1:], ydata[nTrain+1:]
        ###############################################################################
        # Generate sample data

        #y_train=np.reshape(y_train,(nTrain,-1))
        #y_test=np.reshape(y_test,(nTest,-1))
        ###############################################################################

        #print len(X.shape)
        
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_SVR(X,X_train,y_train,X_test,y_test)
        else:

            RMSE=np.apply_along_axis( self.run_SVR,1,X,X_train,y_train,X_test,y_test)

        #print RMSE    
        return RMSE*self.ismax
        
class AlloyCooking_Profiling:
    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,3*3600)),('Time2',(1*3600,3*3600)),('Temp1',(200,300)),('Temp2',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        #print X
        x1=X[0]
        x2=X[1]
        x3=X[2]
        x4=X[3]
        
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.0004056486;# dataset1
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        
        #mycooktemp=matlab.double(np.array([x3,x4]))
        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(X)
            #print  "Strength={:.2f} AveRadius={:.5f} PhaseFraction={:.5f}".format(Strength,AveRad,PhaseFraction)

        else:

            #Strength,AveRad,PhaseFraction=np.apply_along_axis( self.run_Profiling,1,X)
            temp=np.apply_along_axis( self.run_Profiling,1,X)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]

        # maximize strength [0-140], minimize ave radius [0-13] and maximize phase fraction [0 2]


        #utility_score=-AveRad/13+PhaseFraction/2
        utility_score=Strength
        
        #print RMSE    
        #return Strength*self.ismax
        return utility_score    

class AlloyCooking_Profiling_3Steps:
    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,3*3600)),('Time2',(1*3600,3*3600)),('Time3',(1*3600,3*3600)),
                                       ('Temp1',(200,300)),('Temp2',(300,400)),('Temp3',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        #print X
        x1=X[0]
        x2=X[1]
        x3=X[2]
        x4=X[3]
        x5=X[4]
        x6=X[5]
        
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.0004056486;# dataset1
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\KWN_Heat_Treatment',nargout=0)
        
        #mycooktemp=matlab.double(np.array([x3,x4]))
        myCookTemp=matlab.double([x4,x5,x6])
        myCookTime=matlab.double([x1,x2,x3])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(X)
            #print  "Strength={:.2f} AveRadius={:.5f} PhaseFraction={:.5f}".format(Strength,AveRad,PhaseFraction)

        else:

            #Strength,AveRad,PhaseFraction=np.apply_along_axis( self.run_Profiling,1,X)
            temp=np.apply_along_axis( self.run_Profiling,1,X)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]

        # maximize strength [0-140], minimize ave radius [0-13] and maximize phase fraction [0 2]


        #utility_score=-AveRad/13+PhaseFraction/2
        utility_score=Strength
        
        #print RMSE    
        #return Strength*self.ismax
        return utility_score  
        
class AlloyCooking_Profiling2:
    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,4*3600)),('Time2',(0*3600,2*3600)),('Temp1',(200,300)),('Temp2',(300,400))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        print X
        x1=X[0]
        x2=X[1]
        x3=X[2]
        x4=X[3]
        
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.63;
        myxmatrix=0.000675056;# dataset2
        myiSurfen=0.096;
        myfSurfen=1.58e-01;
        myRadsurfenchange=5e-09;
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        
        #mycooktemp=matlab.double(np.array([x3,x4]))
        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength,averad,phasefraction=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange,nargout=3)

        # minimize ave radius [0.5] and maximize phase fraction [0 2]
        temp_str=np.asarray(strength)
        temp_averad=np.asarray(averad)
        temp_phasefrac=np.asarray(phasefraction)
        return temp_str[0][1],temp_averad[0][1],temp_phasefrac[0][1]
        
    def func(self,X):
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            Strength,AveRad,PhaseFraction=self.run_Profiling(X)
            print  "Strength={:.2f} AveRadius={:.5f} PhaseFraction={:.5f}".format(Strength,AveRad,PhaseFraction)
        else:

            #Strength,AveRad,PhaseFraction=np.apply_along_axis( self.run_Profiling,1,X)
            temp=np.apply_along_axis( self.run_Profiling,1,X)
            Strength=temp[:,0]
            AveRad=temp[:,1]
            PhaseFraction=temp[:,2]

        # maximize strength [0-140], minimize ave radius [0-5] and maximize phase fraction [0 2]


        #utility_score=-AveRad/5+PhaseFraction/2
        utility_score=Strength
        #print RMSE    
        #return Strength*self.ismax
        return utility_score

        
class AlloyKWN_Fitting:
    '''
    AlloyKWN_Fitting: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('myEM',(0.6,0.65)),('iSurfen',(0.0958,0.0962)),('fsurfen',(0.06,0.2)),('radsurfenchange',(8e-10,5e-9))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='AlloyKWN_Fitting'
        
    
    def run_Evaluate_KWN(self,X):
        print X
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'P:\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        eng.addpath(r'P:\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting\data',nargout=0)
        eng.addpath(r'P:\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting\BO-matlab-code',nargout=0)

        temp=matlab.double(X.tolist())
        myEM=temp[0][0]
        myiSurfen=temp[0][1]
        myfSurfen=temp[0][2]
        myradchange=temp[0][3]

        RMSE=eng.Evaluating_Alloy_Model_wrt_FourParameters(myEM,myiSurfen,myfSurfen,myradchange)

        return RMSE
        
    def func(self,X):
        X=np.asarray(X)
            
        
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_Evaluate_KWN(X)
        else:

            RMSE=np.apply_along_axis( self.run_Evaluate_KWN,1,X)

        #print RMSE    
        return RMSE*self.ismax


class VPSC7_Fitting:
    '''
    AlloyKWN_Fitting: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 13
        
        if bounds == None: 
            # Theta1>Theta0
            self.bounds = OrderedDict([('PrismaticTau0',(67,77)),('PrismaticTau1',(35,45)),('PrismaticTheta0',(110,10000)),('PrismaticTheta1',(0,100)),
                                       ('BasalTau0',(4,14)),('BasalTau1',(0,6)),('BasalTheta0',(110,10000)),('BasalTheta1',(0,100)),
                                        ('PyramidalTau0',(95,105)),('PyramidalTau1',(95,105)),('PyramidalTheta0',(110,10000)),('PyramidalTheta1',(0,100)),
                                        ('TensileTwiningTau0',(42,52))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='AlloyKWN_Fitting'
        
    
    def run_Fitting_VPSC7(self,X):
        print X
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b',nargout=0)
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b\vu_code',nargout=0)
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b\Wang2010A',nargout=0)
        eng.cd(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b',nargout=0)

        temp=matlab.double(X.tolist())
        Error=eng.VPSC7_Evaluation(temp[0])

        #print Error
        return Error
        
    def func(self,X):
        X=np.asarray(X)
            
        
        if len(X.shape)==1: # 1 data point
            Error=self.run_Fitting_VPSC7(X)
        else:

            Error=np.apply_along_axis( self.run_Fitting_VPSC7,1,X)

        #print RMSE    
        return Error*self.ismax


class VPSC7_Fitting_9Variables:
    '''
    AlloyKWN_Fitting: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 9
        
        if bounds == None: 
            # Theta1>Theta0
            self.bounds = OrderedDict([('PrismaticTau1',(35,45)),('PrismaticTheta0',(110,10000)),('PrismaticTheta1',(0,100)),
                                      ('BasalTau1',(0,6)),('BasalTheta0',(110,10000)),('BasalTheta1',(0,100)),
                                       ('PyramidalTau1',(95,105)),('PyramidalTheta0',(110,10000)),('PyramidalTheta1',(0,100))
                                       ])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='AlloyKWN_Fitting'
        
    
    def run_Fitting_VPSC7_9Variables(self,X):
        print X
        
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b',nargout=0)
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b\vu_code',nargout=0)
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b\Wang2010A',nargout=0)
        eng.cd(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b',nargout=0)

        temp=matlab.double(X.tolist())
        Error=eng.VPSC7_Evaluation_9variables(temp[0])

        return Error
        
    def func(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Error=self.run_Fitting_VPSC7_9Variables(X)
        else:

            Error=np.apply_along_axis( self.run_Fitting_VPSC7_9Variables,1,X)
        return Error*self.ismax

class VPSC7_Fitting_Line46_Thres1:
    '''
    AlloyKWN_Fitting: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 1
        
        if bounds == None: 
            # Theta1>Theta0
            self.bounds = OrderedDict([('Thres1_Line46',(0.4,0.9))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='AlloyKWN_Fitting'
        
    
    def run_Fitting_VPSC7_Line46_Thres1(self,X):
        print X
        
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b',nargout=0)
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b\vu_code',nargout=0)
        eng.addpath(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b\Wang2010A',nargout=0)
        eng.cd(r'F:\Dropbox\05.WithSanSunSvetha\Vu_Sunil_Santu\cp_parameteroptimization\VPSC7b',nargout=0)

        temp=matlab.double(X.tolist())
        Error=eng.VPSC7_Evaluation_line46_thres1(temp[0])

        return Error
        
    def func(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Error=self.run_Fitting_VPSC7_Line46_Thres1(X)
        else:

            Error=np.apply_along_axis( self.run_Fitting_VPSC7_Line46_Thres1,1,X)
        return Error*self.ismax
        
class DeepLearning_MLP_MNIST:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 7
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('n_node1',(100,1000)),('dropout1',(0.01,0.5)),('n_node2',(100,500)),('dropout2',(0.01,0.5)),
                                        ('lr',(0.01,1)),('decay',(1e-8,1e-5)),('momentum',(0.5,1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='DeepLearning_MLP_MNIST'
        
    
    def run_MLP_MNIST(self,X,X_train,Y_train,X_test,Y_test):
        #print X
        # Para: 512, dropout 0.2, 512, 0.2, 10
        
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD, Adam, RMSprop
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 5
        
        model = Sequential()
        x1=np.int(X[0])
        model.add(Dense(x1, input_shape=(784,)))
        
        model.add(Activation('relu'))
        
        temp=np.int(X[1]*100)
        x2=temp*1.0/100
        
        model.add(Dropout(x2))
        #model.add(Dense(512))
        
        x3=np.int(X[2])
        model.add(Dense(x3))
        model.add(Activation('relu'))
        
        temp=np.int(X[3]*100)
        x4=temp*1.0/100
        
        model.add(Dropout(x4))
        
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        #model.summary()
        
        # learning rate, decay, momentum
        sgd = SGD(lr=X[4], decay=X[5], momentum=X[6], nesterov=True)
        
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        return score[1]
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
        
        from keras.datasets import mnist

        from keras.utils import np_utils
    
        X=np.asarray(X)
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 10

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        

        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_MLP_MNIST(X,X_train,Y_train,X_test,Y_test)
        else:

            Accuracy=np.apply_along_axis( self.run_MLP_MNIST,1,X,X_train,Y_train,X_test,Y_test)

        #print RMSE    
        return Accuracy*self.ismax     
        

class DeepLearning_MLP_MNIST_3layers:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 9
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('n_node1',(100,1000)),('dropout1',(0.01,0.5)),('n_node2',(100,500)),('dropout2',(0.01,0.5)),
                                        ('n_node3',(100,200)),('dropout3',(0.01,0.5)),('lr',(0.01,1)),('decay',(1e-8,1e-5)),('momentum',(0.5,1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='DeepLearning_MLP_MNIST'
        
    
    def run_MLP_MNIST(self,X,X_train,Y_train,X_test,Y_test):
        #print X
        # Para: 512, dropout 0.2, 512, 0.2, 10
        
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD, Adam, RMSprop
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 5
        
        model = Sequential()
        x1=np.int(X[0])
        model.add(Dense(x1, input_shape=(784,)))
        
        model.add(Activation('relu'))
        
        temp=np.int(X[1]*100)
        x2=temp*1.0/100
        
        model.add(Dropout(x2))
        #model.add(Dense(512))
        
        x3=np.int(X[2])
        model.add(Dense(x3))
        model.add(Activation('relu'))
        
        temp=np.int(X[3]*100)
        x4=temp*1.0/100
        
        model.add(Dropout(x4))
        
        
        x5=np.int(X[4])
        model.add(Dense(x5))
        model.add(Activation('relu'))
        
        temp=np.int(X[5]*100)
        x6=temp*1.0/100
        
        model.add(Dropout(x6))
        
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        #model.summary()
        
        # learning rate, decay, momentum
        sgd = SGD(lr=X[6], decay=X[7], momentum=X[8], nesterov=True)
        
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        return score[1]
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
        
        from keras.datasets import mnist

        from keras.utils import np_utils
    
        X=np.asarray(X)
        
        nb_classes = 10

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        

        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_MLP_MNIST(X,X_train,Y_train,X_test,Y_test)
        else:

            Accuracy=np.apply_along_axis( self.run_MLP_MNIST,1,X,X_train,Y_train,X_test,Y_test)

        #print RMSE    
        return Accuracy*self.ismax 



class Robot_BipedWalker:

    '''
    Robot Walker: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 8
        
        if bounds == None:  
            self.bounds = OrderedDict([('a1',(0,2)),('a2',(-1,1)),('a3',(-1,1)),('a4',(-6,-3)),
                                        ('a5',(-4,-3)),('a6',(2,4)),('a7',(3,5)),('a8',(-1,2))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='Robot_BipedWalker'
        
    
    def run_BipedWalker(self,X):
        #print X
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\WGCCM_three_link_walker_example\WGCCM_three_link_walker_example',nargout=0)

        temp=matlab.double(X.tolist())


        hz_velocity=eng.walker_evaluation(temp[0])

        if math.isnan(hz_velocity) or math.isinf(hz_velocity):
            hz_velocity=0
        return hz_velocity
        
    def func(self,X):
        
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            velocity=self.run_BipedWalker(X)
        else:

            velocity=np.apply_along_axis( self.run_BipedWalker,1,X)

        return velocity*self.ismax

        
class DeepLearning_CNN_MNIST:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 8
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('nb_filter',(10,50)),('nb_pool',(5,20)),('dropout1',(0.01,0.5)),('dense1',(64,200)),
                                        ('dropout2',(0.01,0.5)),('lr',(0.01,1)),('decay',(1e-8,1e-5)),('momentum',(0.5,1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='DeepLearning_MLP_MNIST'
        
    
    def run_CNN_MNIST(self,X,X_train,Y_train,X_test,Y_test):
        #print X
        # Para: 512, dropout 0.2, 512, 0.2, 10
        
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.optimizers import SGD, Adam, RMSprop
        from keras.layers import Convolution2D, MaxPooling2D

        batch_size = 128 #var1
        nb_classes = 10
        nb_epoch = 1
                
        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = np.int(X[0]) #var1
        #nb_filters = 32
        # size of pooling area for max pooling
        nb_pool = np.int(X[1]) #var2
        #nb_pool = 2
        # convolution kernel size
        kernel_size = (3, 3)
        
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=(1, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        
        temp=np.int(X[2]*100)
        x3=temp*1.0/100
        model.add(Dropout(x3))#var3
        
        
        model.add(Flatten())
        
        temp=np.int(X[3]*100)
        x4=np.int(temp)
        model.add(Dense(x4))#var4        
        model.add(Activation('relu'))
        
        temp=np.int(X[4]*100)
        x5=temp*1.0/100
        model.add(Dropout(x5))#var5
        
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        #model.summary()
        
        # learning rate, decay, momentum
        sgd = SGD(lr=X[5], decay=X[6], momentum=X[7], nesterov=True)
        #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        return score[1]
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
        
        from keras.datasets import mnist

        from keras.utils import np_utils
    
        X=np.asarray(X)
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 1

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        #X_train = X_train.reshape(60000, 784)
        #X_test = X_test.reshape(10000, 784)
        #X_train = X_train.astype('float32')
        #X_test = X_test.astype('float32')
        #X_train /= 255
        #X_test /= 255
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        

        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_CNN_MNIST(X,X_train,Y_train,X_test,Y_test)
        else:
            Accuracy=np.apply_along_axis( self.run_CNN_MNIST,1,X,X_train,Y_train,X_test,Y_test)

        #print RMSE    
        return Accuracy*self.ismax  
        
        
         
class BayesNonMultilabelClassification:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  
            self.bounds = OrderedDict([('eta_xx',(0.0001,0.05)),('eta_yy',(0.000001,0.05)),('svi_rate',(0.000001,0.001)),('lambda',(30,60)),
                                        ('trunc',(0.000001,0.000005)),('alpha',(0.7,1.1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='BNMC'
        
        #import matlab.engine
        #import matlab
        #eng = matlab.engine.start_matlab()
        
        """
        import scipy.io
        mydata=scipy.io.loadmat(r'P:\03.Research\05.BayesianOptimization\PradaBayesianOptimization\run_experiments\run_experiment_unbounded\BNMC\SceneData.mat')
        xxTrain=mydata['xxTrain']
        self.xxTrain=matlab.double(xxTrain.tolist())
        xxTest=mydata['xxTest']
        self.xxTest=matlab.double(xxTest.tolist())
        yyTrain=mydata['yyTrain']
        self.yyTrain=matlab.double(yyTrain.tolist())
        yyTest=mydata['yyTest']
        self.yyTest=matlab.double(yyTest.tolist())
        self.isloaded=1
        """
    def run_BNMC(self,X):
        #print X
        ##########################CHANGE PATH##################################
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\utilities',nargout=0)
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\data',nargout=0)
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC',nargout=0)

        # convert variables
        temp=matlab.double(X.tolist())

        F1score=eng.BayesOpt_BNMC(temp[0])
        #F1score=eng.BayesOpt_BNMC(temp[0],self.xxTrain,self.xxTest,self.yyTrain,self.yyTest)
        #print F1score

        return F1score
        
    def func(self,X):
        #print X
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            F1score=self.run_BNMC(X)
        else:

            F1score=np.apply_along_axis( self.run_BNMC,1,X)

        return F1score*self.ismax 

class BayesNonMultilabelClassificationEnron:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6
        
        if bounds == None:  
            self.bounds = OrderedDict([('eta_xx',(0.0001,0.05)),('eta_yy',(0.000001,0.05)),('svi_rate',(0.000001,0.001)),('lambda',(30,60)),
                                        ('trunc',(0.000001,0.000005)),('alpha',(0.7,1.1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='BNMC_enron'
        
        #import matlab.engine
        #import matlab
        #eng = matlab.engine.start_matlab()
        
        """
        import scipy.io
        mydata=scipy.io.loadmat(r'P:\03.Research\05.BayesianOptimization\PradaBayesianOptimization\run_experiments\run_experiment_unbounded\BNMC\SceneData.mat')
        xxTrain=mydata['xxTrain']
        self.xxTrain=matlab.double(xxTrain.tolist())
        xxTest=mydata['xxTest']
        self.xxTest=matlab.double(xxTest.tolist())
        yyTrain=mydata['yyTrain']
        self.yyTrain=matlab.double(yyTrain.tolist())
        yyTest=mydata['yyTest']
        self.yyTest=matlab.double(yyTest.tolist())
        self.isloaded=1
        """
    def run_BNMC(self,X):
        #print X
        ##########################CHANGE PATH##################################
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\utilities',nargout=0)
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC\data',nargout=0)
        eng.addpath(r'D:\OneDrive\Documents\PhD\Code\Bayesian\PradaBayesianOptimization\real_experiment\BNMC',nargout=0)

        # convert variables
        temp=matlab.double(X.tolist())

        F1score=eng.BayesOpt_BNMC_enron(temp[0])
        #F1score=eng.BayesOpt_BNMC(temp[0],self.xxTrain,self.xxTest,self.yyTrain,self.yyTest)
        #print F1score

        return F1score
        
    def func(self,X):
        #print X
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            F1score=self.run_BNMC(X)
        else:

            F1score=np.apply_along_axis( self.run_BNMC,1,X)

        return F1score*self.ismax 
        
class Alloy_2050_NotSC:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None:  
            self.bounds = OrderedDict([('Cu',(0.02,0.07)),('Li',(0.01,0.06)),('Mg',(0.001,0.005)),('Zr',(0.0005,0.0008))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='Alloy2050'
        
        # flag=0 => init Thermocal, otherwise flag=1
        self.flag=0
        
    
    def run_ThermocalMatlab(self,X):
        #print X
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        ##########################CHANGE PATH##################################
        eng.addpath(r'C:\Users\santurana\Projects\Vu_Experiment',nargout=0)
        eng.addpath(r'C:\Users\santurana\Projects\Matlab-R2015b\toolbox',nargout=0)

        # convert variables
        temp=matlab.double(X.tolist())

        utility=eng.Thermocalc_Alloy_2050_noSC(temp[0],0)
        #print utility

        return utility
        
    def func(self,X):
        
        #print X
        X=np.asarray(X)
            
        if len(X.shape)==1: # 1 data point
            utility=self.run_ThermocalMatlab(X)
        else:
            utility=np.apply_along_axis( self.run_ThermocalMatlab,1,X)

        return utility*self.ismax 
        
        
        
 