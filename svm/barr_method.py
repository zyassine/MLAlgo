import numpy as np
from function import *
  
def Newton(t,x,f,grad,hess):
    #Line search parameters
    alpha = 0.3
    beta = 0.5
    t_0 = 1
    #Computing the gradient, the hessian and the descent direction
    G = grad(x,t)
    H = hess(x,t)
    dx_nt = -np.dot(np.linalg.pinv(H), G)
    #Starting the backtracking line search
    s = t_0
    while f(x+s*dx_nt,t) >= f(x,t) + alpha*s*np.dot(np.transpose(G),dx_nt) and s<=1/np.power(10,8):
        s = beta*s
    #Updating x and computing the upper bound of the error
    x_new = x + s*dx_nt;
    gap = 0.5*np.dot(np.transpose(G), dx_nt)
    return x_new, gap
    

def centering_step(Q,p,A,b,x,t,tol):
    #Creating the function, gradient and hessian
    f = obj_gen(Q,p,A,b)
    grad = grad_gen(Q,p,A,b)
    hess = hess_gen(Q,A,b)
    #Starting Newton's method
    gap = 2*tol
    x_sol = x
    while gap > tol:
        #Computing the Newton step
        x_new,g = Newton(t,x,f,grad,hess);
        x_sol = x_new
        if g==gap:
            break
        gap = g
    return x_sol

def barr_algo(Q,p,A,b,x_0,mu,tol):
    #Computing and initializing the necessary parameters
    n= A.shape[0]
    gap = 2*tol
    t = 1
    x_seq = []
    x = x_0
    #Looping over the centering step
    while gap > tol:
        #Performing the centering step
        xs = centering_step(Q,p,A,b,x,t,tol)
        x = xs
        g = n/t
        if g==gap:
            break
        t = mu*t
        gap = g
    return x     
