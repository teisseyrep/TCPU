import numpy as np
import time
import pandas as pd
from sklearn import preprocessing


from datasets import make_pu_cc_dataset
from TCPU.tcpu import TCPU
from DRPU.drpu import DRPU_custom
from KM.Kernel_MPE import KM2_estimate
from utils import  sample_fixed_class_prior


# Parameters:
c = 0.5
n_sample = 2000
path = 'data/'
n_sym = 20   

ds_seq = ['MNIST','CIFAR10','Fashion']

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
        
                    df_name = path + ds + '_train.csv'
                    df = pd.read_csv(df_name, sep=',')
                    df = df.to_numpy()
                    p = df.shape[1]-1
                    X_train = df[:,0:p]
                    Y_train = df[:,p]
    
                    # Load test data:
                    df_name = path + ds + '_test.csv'
                    df = pd.read_csv(df_name, sep=',')
                    df = df.to_numpy()
                    p = df.shape[1]-1
                    X_test = df[:,0:p]
                    Y_test = df[:,p]    
        
                    if ds=='MNIST':    
                        sel = [441, 151, 177, 145, 364, 407, 368, 199, 507, 324,  91, 410, 456,
                                 386,  18, 189, 179, 158, 213,  25,  65, 139, 155,  58,   8, 436,
                                 327, 439, 219,  49, 321, 109, 208, 496, 478, 266, 203, 204, 483,
                                  82,  28, 206, 226,  60,  11, 183, 446, 260, 133,  59]  
                    elif ds=="CIFAR10":
                        sel = [264,  24, 281, 423, 481, 185, 203, 251, 256,   2, 136, 177, 220,
                               441, 411, 493, 415, 270, 278,  22, 470,  18, 217,  33, 191, 353,
                               131, 110, 143, 298, 194,  62, 427, 112,  70,  86, 182, 496, 291,
                                15, 205, 222, 294, 121,  44, 347,  57,  43, 367, 317]
                    elif ds=="USPS":
                        sel = [247, 242, 375, 184, 416, 298, 493, 345, 281, 322, 321,  88, 425,
                               294, 169, 240,  50, 151, 408,  22, 223, 364, 209,  24,   3, 199,
                               389, 394, 395, 373, 243,  82, 269, 510, 116, 312, 254,  98, 309,
                                74, 205, 286,  59, 315,  38,  11, 249, 437, 180, 409]    
                    elif ds=="Fashion":
                        sel = [ 95,  76, 335, 439, 259, 446, 199, 444,  72, 448, 285, 426, 292,
                               423,  89, 424,  45, 276, 430, 314, 437,  86, 288, 219,  16, 143,
                               458, 410,   3, 445, 185, 309, 451, 102,  98, 313, 132, 453,  39,
                               418, 471, 171, 184, 197, 400, 428, 507, 469, 352, 328]    
                    elif ds=="KMNIST":
                        sel = [326,  42, 485, 344, 115, 139, 438, 121, 323,  28, 445, 306, 500,
                               501, 172, 232, 409, 442, 483,  10, 106, 339, 219, 509, 220, 123,
                               331, 298, 424, 155,  31, 133, 151, 254, 209, 481, 132, 397, 120,
                               207, 239,  46, 240, 415,  12, 223,  18, 472, 138,  55]  
    
                    sel = sel[0:30] #select 30 most relevant features
                    
                    X_train = X_train[:,sel]
                    X_test = X_test[:,sel]
                    scaler = preprocessing.StandardScaler().fit(X_train)
                    X_train = scaler.transform(X_train)
                    scaler = preprocessing.StandardScaler().fit(X_test)
                    X_test = scaler.transform(X_test)
    
    
                    X_train, Y_train = sample_fixed_class_prior(X_train,Y_train,pi=pi_train,sym=sym)
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
        
            np.savetxt('results_images/exp1_error_' + method  + '_' + ds + '_c' + str(c)  +'_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +  '.txt', res_error, fmt='%1.3f')
            np.savetxt('results_images/exp1_time_' + method  + '_' + ds + '_c' + str(c) +'_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +  '.txt', res_time, fmt='%1.3f')
            np.savetxt('results_images/exp1_est_' + method  + '_' + ds + '_c' + str(c)  +'_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +  '.txt', res_est, fmt='%1.3f')
    
    
         
        
    
















