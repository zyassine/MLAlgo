import numpy as np

def obj_gen(Q,p,A,b):
    def f(x,t):
        el1 = -np.sum(np.log(np.absolute(b-np.dot(A,x))))
        el2 = t*(0.5*np.dot(np.transpose(x), np.dot(Q,x)) + np.dot(np.transpose(p),x))
        return el1 + el2
    return f 

def grad_gen(Q,p,A,b):
    def gradient(x,t):
        el1 = t*(np.dot(Q,x) +p)
        v = b-np.dot(A,x)
        el2 = np.sum(np.dot(np.transpose(A), np.reciprocal(v)))
        return el1 + el2
    return gradient

def hess_gen(Q,A,b):
    def hess(x,t):
        hessian = t*Q
        mat = np.zeros(hessian.shape)
        v = np.square(b-np.dot(A,x))
        for i in range(mat.shape[0]):
            for j in range(i,mat.shape[1]):
                s = 0
                for k in range(A.shape[1]):
                    s += A[k,i]*A[k,j]/v[i]
                    mat[i,j] = s
                    mat[j,i] = s    
        return hessian + mat
    return hess