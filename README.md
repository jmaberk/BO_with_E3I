# BO_with_E3I

This code runs the Exploration Enhanced Expected Improvement acquisition functions against other common acquisition functions. The paper describing the algorithm has been submitted for review to the ECML 2018 conference.
NOTE: The code here is functional but actively being commented and updated to make it easier to understand and use.


## System Requirements
This code requires python 2.7. It will need to be modified to work with later versions of python. It requires several standard python packages such as numpy, scipy, pickle, itertools, random, seaborn, matplotlib, sklearn, math, time, mpl_toolkits, and copy. 

The real world experiments require Matlab engine (https://au.mathworks.com/help/matlab/matlab-engine-for-python.html) and Keras (https://keras.io/). Keras itself requires TensorFlow, CNTK, or Theano. The rest of the code does not require these so it can be run without them by removing any imports of real_experiment_function.py.

We used the code on a windows 10 machine with Matlab R2015b and a Theano-based Keras.

## Previous work
Some code, including the real-world functions, are taken from previous works by Nguyen et al. These can be found here: https://github.com/ntienvu/ICDM2017_FBO and here: https://github.com/ntienvu/ICDM2016_B3O

## Example code
A quick example of the code is shown in example.py. This shows the optimization of the two peak Gaussian mixture discussed in the paper.

## Usage
IMPORTANT: The pickle_location variable in bayesianOptimizationMaster.py must be changed to the location of your pickleStorage file for this code to run

The file bayesianOptimizationMaster.py controls the rest of the code It is set up to run a basic 2D Gaussian mixture but instructions running other functions and altering other parameters are given as comments in the code between rows of hash symbols.
