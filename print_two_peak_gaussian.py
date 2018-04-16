'''
Name: example.py
Authors: Julian Berk and Vu Nguyen
Publication date:16/04/2018
Inputs:None
Outputs: A plot of the 1D Gaussian mixture used in the paper
'''

import numpy as np
from collections import OrderedDict
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prada_bayes_opt import functions
class plot_two_peak_gaussian(object):
    def __init__(self):
        fig=plt.figure(figsize=(10, 7))
        f1=functions.doubleGaussian(dim=1)
        x=np.arange(0,1,0.001)
        y=f1.func(x).reshape(x.shape)
        plt.plot(x,y)
        strTitle="Plot of the Two Peak Gaussian Mixture in 1D"
        plt.title(strTitle,fontdict={'size':22})
        strFile="Gaussian_mixture_shape.pdf"
        plt.savefig(strFile, bbox_inches='tight')
        plt.ylabel('f(x)',fontdict={'size':18})
        #plt.xlabel('Number of Evaluations')
        plt.xlabel('x',fontdict={'size':18})
        plt.savefig(strFile, bbox_inches='tight')