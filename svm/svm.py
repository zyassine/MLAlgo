import numpy as np 


def transform_svm_dual(C,X,y):
	n = X.shape[0]
	xtx = np.dot(X,np.transpose(X))
	xtxy = np.dot(xtx, np.diagflat(y))        
	Q = np.dot(np.diagflat(y),xtxy)        
	p = -np.ones((n,1))
	A = np.concatenate( (np.identity(n), -np.identity(n)))
	b = np.concatenate((C*np.ones((n,1)), np.zeros((n,1))))
	return Q,p,A,b		