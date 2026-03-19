import numpy as np
import time

from datasets import generate_artificial_data, make_pu_cc_dataset
from TCPU.tcpu import TCPU


# Parameters:
method = "TCPU"
ds = 'art1'
c = 0.5
n_sample = 2000
n_sym = 20

#Case 1:
pi_train = 0.1
pi_test = 0.8

#Case 2:
#pi_train = 0.2
#pi_test = 0.8

#Case 3:
#pi_train = 0.5
#pi_test = 0.8


#Case 4:
#pi_train = 0.1
#pi_test = 0.5

#Case 5:
#pi_train = 0.2
#pi_test = 0.5

#Case 6:
pi_train = 0.5
pi_test = 0.5

if pi_train==0.1:
    hat_pi_train_seq = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
if pi_train==0.2:
    hat_pi_train_seq = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4])
if pi_train==0.5:
    hat_pi_train_seq = np.array([0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7])


res_error = np.zeros((n_sym,len(hat_pi_train_seq)))
res_est = np.zeros((n_sym,len(hat_pi_train_seq)))

counter = 0
for hat_pi_train in hat_pi_train_seq: 
    print('\n --- hat_pi_train = ' + str(hat_pi_train) + ' --- \n')
    
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
            hat_pi_train, hat_pi_test = TCPU(X_train_pos,X_train_un,X_test,pi_train=hat_pi_train)
        
        
        end_time = time.time()
        run_time = end_time - start_time
        

        res_error[sym,counter] = np.abs((hat_pi_test-pi_test))
        res_est[sym,counter] = hat_pi_test

    counter = counter + 1

    np.savetxt('results_artificial/exp3_error_' + method  + '_' + ds + '_c' + str(c) + '_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +   '.txt', res_error, fmt='%1.3f')
    np.savetxt('results_artificial/exp3_est_' + method  + '_' + ds + '_c' + str(c) + '_n_sample' + str(n_sample) + '_pi_train' + str(pi_train) +   '.txt', res_est, fmt='%1.3f')
    
    
    
     
    
    


















