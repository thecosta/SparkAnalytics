# -*- coding: utf-8 -*-
import sys
import argparse
import functools
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext


def readData(input_file,spark_context):
    """  Read data from an input file and return rdd containing pairs of the form:
	                 (x,y)
	 where x is a numpy array and y is a real value. The input file should be a 
	 'comma separated values' (csv) file: each line of the file should contain x
         followed by y. For example, line:

         1.0,2.1,3.1,4.5

         should be converted to tuple:
	
         (array(1.0,2.1,3.1),4.5)
	 

    """ 
    return spark_context.textFile(input_file)\
		.map(lambda line:line.split(','))\
		.map(lambda words:(words[:-1],words[-1]))\
		.map(lambda (features,target): (np.array([ float(x) for x in features]),float(target)))


def readBeta(input):
    """ Read a vector β from CSV file input
    """
    with open(input,'r') as fh:
        str_list = fh.read()\
                   .strip()\
		   .split(',')
        return np.array( [float(val) for val in str_list] )           


def writeBeta(output,beta):
    """ Write a vector β to a CSV file ouptut
    """
    with open(output,'w') as fh:
	fh.write(','.join(map(str, beta.tolist()))+'\n')
    

def estimateGrad(fun,x,delta):
     """ Given a real-valued function fun, estimate its gradient numerically.
     """
     d = len(x)
     grad = np.zeros(d)
     for i in range(d):
         e = np.zeros(d)
         e[i] = 1.0
         grad[i] = (fun(x+delta*e) - fun(x))/delta
     return grad

 
def lineSearch(fun,x,grad,a=0.2,b=0.6):
    """ Given function fun, a current argument x, and gradient grad, 
	perform backtracking line search to find the next point to move to.
	(see Boyd and Vandenberghe, page 464).

	Parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient  ∇fun(x), the function finds a t such that
	fun(x - t * grad) <= fun(x) - a t <∇fun(x),∇fun(x)>

	The return value is the resulting value of t.
    """
    t = 1.0
    while fun(x-t*grad) > fun(x)- a * t *np.dot(grad,grad):
	t = b * t
    return t     


def predict(x,beta):
    """ Given vector x containing features and parameter vector β, 
	return the predicted value: 

	                y = <x,β>   

    """
    return np.dot(x,beta)


def f(x,y,beta):
    """ Given vector x containing features, true label y, 
	and parameter vector β, return the square error:

	         f(β;x,y) =  (y - <x,β>)^2	

    """
    return (y - predict(x,beta))**2

def localGradient(x,y,beta):
    """ Given vector x containing features, true label y, 
	and parameter vector β, return the gradient ∇f of f:

	        ∇f(β;x,y) =  -2 * (y - <x,β>) * x	

        with respect to parameter vector β.

        The return value is  ∇f.
    """
    return -2*(y - predict(x,beta))*x

 
def F(data,beta,lam = 0):
    """  Compute the regularized mean square error:

             F(β) = 1/n Σ_{(x,y) in data}    f(β;x,y)  + λ ||β ||_2^2   
                  = 1/n Σ_{(x,y) in data} (y- <x,β>)^2 + λ ||β ||_2^2 

         where n is the number of (x,y) pairs in RDD data. 

	 Inputs are:
            - data: an RDD containing pairs of the form (x,y)
            - beta: vector β
	    - lam:  the regularization parameter λ
           

	 The return value is F(β).

    """
    n = data.count()
    r = data.map(lambda (x,y): f(x,y,beta)) \
            .reduce(lambda x,y: x+y)
    return r/n + lam*np.sum(beta**2)


def gradient(data,beta,lam = 0):
    """ Compute the gradient  ∇F of the regularized mean square error 
                F(β) = 1/n Σ_{(x,y) in data} f(β;x,y) + λ ||β ||_2^2   
                     = 1/n Σ_{(x,y) in data} (y- <x,β>)^2 + λ ||β ||_2^2   
 
	where n is the number of (x,y) pairs in data. 

	Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β
	     - lam:  the regularization parameter λ
             

        The return value is an array containing ∇F.

    """
    n = data.count()
    r = data.map(lambda (x,y): localGradient(x,y,beta)) \
            .reduce(lambda x,y: x+y)
    return r/n + 2*lam*beta


def test_square_error(sc):
    '''Test square error of localGradient.'''
    y = 1.0
    x = np.array([np.cos(t) for t in range(5)])
    beta = np.array([np.sin(t) for t in range(5)])
    
    def evaluate_f(beta):
        return f(x,y,beta)
    
    print('localGradient: %s' % localGradient(x,y,beta))
    print('estimateGrad: %s' % estimateGrad(evaluate_f, beta, 0.0000001))
    

def test_ridge_regression(sc):
    '''Test ridge regression gradient.'''

    data = readData('data/small.test', sc)
    beta = np.array([np.sin(t) for t in range(50)])
    lam = 1.0
    
    def evaluate_F(beta):
        return F(data, beta, lam)
    
    print('gradient: %s' % gradient(data, beta, lam))
    print('estimateGrad: %s' % estimateGrad(evaluate_F, beta, 0.000000001)) 
    
     
def test(data,beta):
    """ Compute the mean square error  
		 MSE(β) =  1/n Σ_{(x,y) in data} (y- <x,β>)^2
        of parameter vector β over the dataset contained in RDD data, where n is the size of RDD data.
	Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β
	The return value is MSE(β).   
    """
    m = data.count()
    rdd = data.map(lambda (x,y): f(x,y,beta)) \
              .reduce(lambda x,y: x+y)
    return rdd/m


def train(data,beta_0,lam,max_iter,eps):
    """ Perform gradient descent:
        to  minimize F given by
             F(β) = 1/n Σ_{(x,y) in data} f(β;x,y) + λ ||β ||_2^2   
	where
             - data: an rdd containing pairs of the form (x,y)
             - beta_0: the starting vector β
	     - lam:  is the regularization parameter λ
             - max_iter: maximum number of iterations of gradient descent
             - eps: upper bound on the l2 norm of the gradient
             - a,b: parameters used in backtracking line search
	The function performs gradient descent with a gain found through backtracking
        line search. That is it computes
	           β_k+1 = β_k - γ_k ∇F(β_k) 
	where the gain γ_k is given by
		  γ_k = lineSearch(F,β_κ,∇F(β_k))
	and terminates after max_iter iterations or when ||∇F(β_k)||_2<ε.   
	The function returns:
	     -beta: the trained β, 
	     -gradNorm: the norm of the gradient at the trained β, and
             -k: the number of iterations performed
    """
    beta_curr = beta_0

    def train_F(beta):
        return F(data, beta, lam)
    
    def norm_grad(grad):
        return np.sum(grad**2)**0.5
    
    start_time = time()
    for k in range(max_iter):
        beta_next = beta_curr - lineSearch(train_F, beta_curr, gradient(data, beta_curr, lam)) * gradient(data, beta_curr, lam)
        euclidean_norm = norm_grad(gradient(data, beta_next, lam))

        if euclidean_norm < eps:
            print('k: %s\ttime: %s\tF(beta): %s\tnorm F: %s' %
                 (k, (time()-start_time), train_F(beta_curr), euclidean_norm))
            return beta_next, euclidean_norm, k
        else:
            beta_curr = beta_next

        print('k: %s\ttime: %s\tF(beta): %s\tnorm F: %s' % 
             (k, (time()-start_time), train_F(beta_curr), euclidean_norm))
    
    return beta_next, euclidean_norm, max_iter
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Parallel Ridge Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model')
    parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model')
    parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
    parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
    parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
    parser.add_argument('--eps', type=float, default=0.01, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.') 
    parser.add_argument('--N',type=int,default=2,help='Level of parallelism')


    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    args = parser.parse_args()
  
    sc = SparkContext(appName='Parallel Ridge Regression')
    
    if not args.verbose :
        sc.setLogLevel("ERROR")	
  
    beta = None

    if args.traindata is not None:
        # Train a linear model β from data with regularization parameter λ, and store it in beta
        print 'Reading training data from',args.traindata
        data = readData(args.traindata,sc)
        data = data.repartition(args.N).cache()
      
        x,y = data.take(1)[0]
        beta0 = np.zeros(len(x))

	print 'Training on data from',args.traindata,'with λ =',args.lam,', ε =',args.eps,', max iter = ',args.max_iter
        beta, gradNorm, k = train(data,beta_0=beta0,lam=args.lam,max_iter=args.max_iter,eps=args.eps) 
	print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
	print 'Saving trained β in',args.beta
        writeBeta(args.beta,beta)
        
     
    if args.testdata is not None:
        # Read beta from args.beta, and evaluate its MSE over data
        print 'Reading test data from',args.testdata
        data = readData(args.testdata,sc)
        data = data.repartition(args.N).cache()
      
        print 'Reading beta from',args.beta
	beta = readBeta(args.beta)

	print 'Computing MSE on data',args.testdata
        MSE = test(data,beta)
	print 'MSE is:', MSE 
	