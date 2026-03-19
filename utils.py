import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def sigma(x):
    res = np.exp(x)/(1+np.exp(x))
    return res


def remove_collinear(X,threshold=0.9): 

    X = pd.DataFrame(X)
    X_corr = X.corr()
    
    columns = np.full((X_corr.shape[0],), True, dtype=bool)
    for i in range(X_corr.shape[0]):
        for j in range(i+1, X_corr.shape[0]):
            if X_corr.iloc[i,j] >= threshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = X.columns[columns]
    return np.array(selected_columns)

def make_binary_class(y):
   if np.unique(y).shape[0]>2: 
       values, counts = np.unique(y, return_counts=True)
       ind = np.argmax(counts)
       major_class = values[ind]
       for i in np.arange(y.shape[0]):
           if y[i]==major_class:
               y[i]=1
           else:
               y[i]=0    
   return y           


def mi_filter(X,y,pmax=50):
    mi = np.zeros(X.shape[1])
    for j in np.arange(X.shape[1]): 
        #print(j)
        mi[j] = mutual_info_classif(X[:,j].reshape(-1,1), y)
    sel = np.argsort(-mi)[0:pmax]
    return sel

def sample_fixed_class_prior(X_train,Y_train,pi=0.1,sym=1):

    """
    Downsampling: the function samples a subset of the give dataset  such that P(Y=1)=pi in the output data.

    """    

    if pi<=np.mean(Y_train):
        w0 = np.where(Y_train==0)[0]
        n0  = len(w0)
        sel_pos = int((pi/(1-pi)) * n0)
        w1 = np.where(Y_train==1)[0]
        np.random.seed(sym)
        w1sel = np.random.choice(w1, size=sel_pos, replace=False)
        sel = np.concatenate((w0, w1sel),0)
        X_train_samp = X_train[sel,:]
        Y_train_samp= Y_train[sel]
    else:
        w1 = np.where(Y_train==1)[0]
        n1  = len(w1)
        sel_neg = int(((1-pi)/pi) * n1)
        w0 = np.where(Y_train==0)[0]
        np.random.seed(sym)
        w0sel = np.random.choice(w0, size=sel_neg, replace=False)
        sel = np.concatenate((w0sel, w1),0)
        X_train_samp = X_train[sel,:]
        Y_train_samp= Y_train[sel]
    
    return X_train_samp, Y_train_samp


def moving_average(x):
    n=x.shape[0]
    x_new = np.zeros(n)
    for i in np.arange(n):
        if i==0 or i==n-1:
            x_new[i] = x[i]
        else:
            x_new[i] = (x[i-1]+x[i]+x[i+1])/3
    
    return x_new