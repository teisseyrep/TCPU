import numpy as np
import time

from datasets import generate_artificial_data, make_pu_cc_dataset
from DRPU.drpu import DRPU_custom
from TCPU.tcpu import TCPU
from KM.Kernel_MPE import KM2_estimate


# Parameters:
ds = 'art1'
c = 0.25
n_sample = 2000
n_sym = 20

for pi_train in np.array([0.05,0.1,0.2,0.5]):

    pi_test_seq = np.array([0.2,0.4,0.6,0.8])
    
    method_seq = np.array(['TCPU','KM2-LS','DRPU'])
    
    
    for method in method_seq:
        print('\n === Method: ' + method + ' === \n')
        
        res_error = np.zeros((n_sym,len(pi_test_seq)))
        res_est = np.zeros((n_sym,len(pi_test_seq)))
        res_time = np.zeros((n_sym,len(pi_test_seq)))
        
        counter = 0
        for pi_test in pi_test_seq: 
            print('\n --- pi_test = ' + str(pi_test) + ' --- \n')
            
            for sym in np.arange(n_sym):
                #print("|",end='')
    
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

            np.savetxt('results_artificial/exp1_error_' + method  + '_' + ds + '_c' + str(c) + '_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +   '.txt', res_error, fmt='%1.3f')
            np.savetxt('results_artificial/exp1_est_' + method  + '_' + ds + '_c' + str(c) + '_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +   '.txt', res_est, fmt='%1.3f')
            np.savetxt('results_artificial/exp1_time_' + method  + '_' + ds + '_c' + str(c) + '_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +  '.txt', res_time, fmt='%1.3f')
            
    
    
     
    
    
















