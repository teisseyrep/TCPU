import os
import numpy as np
import time

from datasets import generate_artificial_data, make_pu_cc_dataset
from DRPU.drpu import DRPU_custom
from TCPU.tcpu import TCPU
from KM.Kernel_MPE import KM2_estimate


# Parameters:
ds = 'art1'
c = 0.5
n_sym = 20

pi_train = 0.1
pi_test = 0.8

#pi_train = 0.2
#pi_test = 0.8

#pi_train = 0.3
#pi_test = 0.8

#pi_train = 0.5
#pi_test = 0.8



method_seq = np.array(['TCPU','KM2-LS','DRPU'])


n_sample_seq = np.array([100,200,300,400,500,750,1000,1250,1500,1750,2000])


for method in method_seq:
    print('\n === Method: ' + method + ' === \n')
    
    res_error = np.zeros((n_sym,len(n_sample_seq)))
    res_est = np.zeros((n_sym,len(n_sample_seq)))
    res_time = np.zeros((n_sym,len(n_sample_seq)))
    
    counter = 0
    for n_sample in n_sample_seq: 
        print('\n --- n_sample= ' + str(n_sample) + ' --- \n')
        
        for sym in np.arange(n_sym):
            print("|",end='')

    
            sym_test = sym  + 5000    

            if ds=="art1":
                Y_train, X_train = generate_artificial_data(20000,p=10,pi=pi_train,a=1,sym=sym)
                Y_test, X_test = generate_artificial_data(n_sample,p=10,pi=pi_test,a=1,sym=sym_test)
                X_train_pos, X_train_un = make_pu_cc_dataset(Y_train,X_train,pi_train=pi_train,n_sample=n_sample,c=c,sym=sym)
    
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

        np.savetxt('results_artificial/exp2_error_' + method  + '_' + ds + '_c' + str(c) + '_pi_test' + str(pi_test) + '_pi_train' + str(pi_train) +   '.txt', res_error, fmt='%1.3f')
        np.savetxt('results_artificial/exp2_est_' + method  + '_' + ds + '_c' + str(c) + '_pi_test' + str(pi_test) + '_pi_train' + str(pi_train) +   '.txt', res_est, fmt='%1.3f')
        np.savetxt('results_artificial/exp2_time_' + method  + '_' + ds + '_c' + str(c) + '_pi_test' + str(pi_test) + '_pi_train' + str(pi_train) +  '.txt', res_time, fmt='%1.3f')
        


     
    
    
















