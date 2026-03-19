import os
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


from datasets import make_pu_cc_dataset
from TCPU.tcpu import TCPU
from DRPU.drpu import DRPU_custom
from KM.Kernel_MPE import KM2_estimate
from utils import  make_binary_class, mi_filter, remove_collinear, sample_fixed_class_prior


ds_seq = ["Diabetes","Spambase","Waveform-5000","Segment","Vehicle","Yeast","Banknote-authentication"]

c = 0.25
path = 'data/'
n_sym= 20


for ds in ds_seq:
    print("=== Dataset",ds," ===\n")
    
    for pi_train in np.array([0.05,0.1,0.2,0.5]):
      
    
        pi_test_seq = np.array([0.2,0.4,0.6,0.8])

        
        method_seq = np.array(['TCPU','KM2-LS','DRPU'])
        
        for method in method_seq:
            print('\n === Method: ' + method + ' === \n')
            
            res_error = np.zeros((n_sym,len(pi_test_seq)))
            res_time = np.zeros((n_sym,len(pi_test_seq)))
            res_est = np.zeros((n_sym,len(pi_test_seq)))
            
            counter = 0
            for pi_test in pi_test_seq: 
                print('\n --- pi_test= ' + str(pi_test) + ' --- \n')
                
                for sym in np.arange(n_sym):
                    #print("|",end='')
        
                    df_name = path + ds + '.csv'
                    df = pd.read_csv(df_name, sep=',')
                    del df['BinClass']
                    df = df.to_numpy()
                    p = df.shape[1]-1
                    Xall = df[:,0:p]
                    yall = df[:,p]
                    yall = make_binary_class(yall)
                    selected_columns = remove_collinear(Xall,0.95)
                    Xall = Xall[:,selected_columns]
                    sel = mi_filter(Xall,yall,pmax=30)
                    Xall = Xall[:,sel]
      
                    X_train, X_test, Y_train, Y_test = train_test_split(Xall, yall, test_size=0.5, random_state=sym)
      
      
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train = scaler.transform(X_train)
                    scaler = preprocessing.StandardScaler().fit(X_test)
                    X_test = scaler.transform(X_test)
    
    
                    X_train, Y_train = sample_fixed_class_prior(X_train,Y_train,pi=pi_train,sym=sym)
                    n_sample = X_train.shape[0]
                    X_train_pos, X_train_un = make_pu_cc_dataset(Y_train,X_train,pi_train=pi_train,n_sample=n_sample,c=c,sym=sym)
                    X_test, Y_test = sample_fixed_class_prior(X_test,Y_test,pi=pi_test,sym=sym)
                    if X_test.shape[0]>n_sample:
                        np.random.seed(sym)
                        samp = np.random.choice(np.arange(X_test.shape[0]), size=n_sample)
                        X_test = X_test[samp,:]
                        Y_test = Y_test[samp]
                    
        
                    start_time = time.time()
                   
        
                    # TCPU:
                    if method=='TCPU':    
                        hat_pi_train, hat_pi_test = TCPU(X_train_pos,X_train_un,X_test,pi_train=None)
                    
                    # DRPU:
                    elif method=='DRPU':
                        X_pos = X_train_pos
                        X_un = X_train_un
                        acc, hat_pi_train, hat_pi_test = DRPU_custom(X_pos,X_un,X_test,Y_test,max_epochs = 200,batch_size = 100,lr = 2e-5,device_num=0,pi_train=None)
                   
                    # KM2-LS:
                    elif method=='KM2-LS':
                        hat_pi_test = KM2_estimate(X_train_pos, X_test)
                    
                    end_time = time.time()
                    run_time = end_time - start_time
                    
        
                    res_error[sym,counter] = np.abs((hat_pi_test-pi_test))
                    res_time[sym,counter] = run_time
                    res_est[sym,counter] = hat_pi_test
        
                counter = counter + 1
        
            np.savetxt('results_bench/exp1_error_' + method  + '_' + ds + '_c' + str(c)  + '_pi_train' + str(pi_train) +  '.txt', res_error, fmt='%1.3f')
            np.savetxt('results_bench/exp1_time_' + method  + '_' + ds + '_c' + str(c) + '_pi_train' + str(pi_train) +  '.txt', res_time, fmt='%1.3f')
            np.savetxt('results_bench/exp1_est_' + method  + '_' + ds + '_c' + str(c)  + '_pi_train' + str(pi_train) +  '.txt', res_est, fmt='%1.3f')
    
    
     
    
    
















