from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from KM.Kernel_MPE import KM2_estimate

def estimate_pi_test(D,E,K,L,M,n,m,nt,pi_train):
    
    a =  (1/(n**2)) * D  
    b = -pi_train * (1/(n*m)) * K
    c = -(1-pi_train) * (1/(n*nt)) * L
    d = -(1/(n*m)) * K
    e = pi_train * (1/(m**2)) * E
    f = (1-pi_train) * (1/(nt*m)) * M
    
    g = (1/(n**2)) * D
    h = - 2 * (1/(n*m)) * K
    j = (1/(m**2)) * E
    
    num = a+b+c+d+e+f
    den = g+h+j
    
    if den==0:
        den = den + 0.0001
    
    res = num/den
    
    return(res)
    



def TCPU(X_train_pos,X_train_un,X_test,pi_train=None,gamma=None,switch=False,switch_thr=0.02):
    
    D  = np.sum( pairwise_kernels(X_train_un, metric='rbf',gamma=gamma) )
    E  = np.sum( pairwise_kernels(X_train_pos, metric='rbf',gamma=gamma) ) 
    #F  = np.sum( pairwise_kernels(X_test, metric='rbf') )
    K  = np.sum( pairwise_kernels(X_train_un, X_train_pos, metric='rbf',gamma=gamma) )
    L  = np.sum( pairwise_kernels(X_train_un, X_test, metric='rbf',gamma=gamma) )
    M  = np.sum( pairwise_kernels(X_train_pos, X_test, metric='rbf',gamma=gamma) )


    n = X_train_un.shape[0]
    m = X_train_pos.shape[0]
    nt = X_test.shape[0] 
    
    if pi_train==None:
        hat_pi_train = KM2_estimate(X_train_pos, X_train_un)
    else:
        hat_pi_train = pi_train

    if(switch):
        g = (1/(n**2)) * D
        h = - 2 * (1/(n*m)) * K
        j = (1/(m**2)) * E
        den = g+h+j
        if den>switch_thr:
            hat_pi_test = estimate_pi_test(D,E,K,L,M,n,m,nt,hat_pi_train)
        else:
            hat_pi_test = KM2_estimate(X_train_pos, X_test)
    else:
        hat_pi_test = estimate_pi_test(D,E,K,L,M,n,m,nt,hat_pi_train)
    
    if hat_pi_test>=1:
        hat_pi_test
        
    if hat_pi_test<=0:
        hat_pi_test=0
    
    return hat_pi_train, hat_pi_test

def kl_dist_kernel(g,D,E,F,K,L,M,n,m,nt,pi_train):


    a2 = ( (1-g)**2 ) * (1/(n**2)) * D
    b2 = ( (1-g)**2 ) * (pi_train**2) * (1/(m**2)) * E
    c2 = ((1-pi_train)**2) * (1/(nt**2)) * F
    d2 = ((1-pi_train)**2) * (g**2) * (1/(m**2)) * E
    
    ab =  ( (1-g)**2 ) * pi_train * (1/(n*m)) * K
    ac = (1-g) * (1-pi_train) * (1/(n*nt)) * L
    ad = (1-g) * (1-pi_train) * g * (1/(n*m)) * K
    bc = (1-g) * (1-pi_train) * pi_train * (1/(m*nt)) * M
    bd = (1-g)*(1-pi_train) * g * pi_train *(1/(m**2)) * E
    cd = ((1-pi_train)**2) * g * (1/(nt*m)) * M
    
    res = a2 + b2 + c2 + d2 + 2*(ab + ac + ad + bc + bd + cd)
    
    return(res)

def dist_p_ppos(X_train_pos,X_train_un,gamma=None):

  D  = np.sum( pairwise_kernels(X_train_un, metric='rbf',gamma=gamma) )
  E  = np.sum( pairwise_kernels(X_train_pos, metric='rbf',gamma=gamma) ) 
  K  = np.sum( pairwise_kernels(X_train_un, X_train_pos, metric='rbf',gamma=gamma) )
 
  n = X_train_un.shape[0]
  m = X_train_pos.shape[0] 
  
  
  g = (1/(n**2)) * D
  h = - 2 * (1/(n*m)) * K
  j = (1/(m**2)) * E
   
  den = g+h+j

  return den
