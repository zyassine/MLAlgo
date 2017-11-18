import numpy as np 

def generate_data(mu0,mu1,sigma0,sigma1,n0,n1):
	x0 = np.random.multivariate_normal(mu0, sigma0, n0)
	x1 = np.random.multivariate_normal(mu1, sigma1, n1)
	X = np.concatenate((x0, x1))
	y = np.concatenate((np.ones((n0,1)), -np.ones((n1,1))))
	return X, y