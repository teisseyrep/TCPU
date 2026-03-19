import numpy as np

def generate_artificial_data(n,p,pi=0.5,a=1,sym=1):

    """
    Generate artificial dataset
    
    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of features in X
    pi_p : int, optional
        P(Y=1)    
    a : int, optional
        difference in means
        
    """
    
    # Generate Y:
    np.random.seed(sym)    
    Y = np.random.binomial(1, pi, size=n)
    X = np.zeros((n,p))
    mu0 = np.zeros(p)
    mu1 = a * np.ones(p)
    
    # Generate X:
    Sigma = np.diag(np.ones(p))
    for i in np.arange(n):
          base = sym*n
        
          if Y[i]==0:
              np.random.seed(base+i) 
              X[i,:] = np.random.multivariate_normal(mean=mu0, cov=Sigma, size=1)
          else:
              np.random.seed(base+i)
              X[i,:] = np.random.multivariate_normal(mean=mu1, cov=Sigma, size=1)
    
    
    return Y, X


def make_pu_cc_dataset_simple(Y_train,X_train,n_pos=500,n_un=500,sym=1):

    n = Y_train.shape[0]
    w1 = np.where(Y_train==1)[0]
    n1 = len(w1)
    np.random.seed(sym) 
    samp1 = np.random.choice(np.arange(n1), size=n_pos, replace=False)
    pos = w1[samp1]
    
    np.random.seed(500+sym) 
    un = np.random.choice(np.arange(n), size=n_un, replace=False)
    X_train_pos = X_train[pos,:]
    X_train_un = X_train[un,:]
    
    return X_train_pos, X_train_un

def make_pu_cc_dataset(Y_train,X_train,pi_train,n_sample=1000,c=0.5,sym=1):

    A = 1/(1-c*(1-pi_train))
    n_pos = int(A*c*pi_train*n_sample)
    n_un = int(A * (1-c) * n_sample)
    
    n = Y_train.shape[0]
    w1 = np.where(Y_train==1)[0]
    n1 = len(w1)
    np.random.seed(sym) 
    samp1 = np.random.choice(np.arange(n1), size=n_pos, replace=False)
    pos = w1[samp1]
    
    np.random.seed(500 + sym) 
    un = np.random.choice(np.arange(n), size=n_un, replace=False)
    X_train_pos = X_train[pos,:]
    X_train_un = X_train[un,:]
    
    return X_train_pos, X_train_un


def label_transform_MNIST(y):
    w1 = (y%2==0)
    w0 = (y%2!=0)
    y[w1]=1
    y[w0]=0
    return y

def label_transform_CIFAR10(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 8, 9]:
            y[i]=1
        else:
            y[i]=0
    return y
   
def label_transform_KMNIST(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 8, 9]:
            y[i]=1
        else:
            y[i]=0
    return y    
   
def label_transform_USPS(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 1, 2, 3, 4]:
            y[i]=1
        else:
            y[i]=0
    return y
 
def label_transform_Fashion(y):
    n = len(y)
    for i in np.arange(0,n):
        if y[i] in [0, 2, 3, 4, 6]:
            y[i]=1
        else:
            y[i]=0
    return y   
    
    



