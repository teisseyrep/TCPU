from datasets import generate_artificial_data, make_pu_cc_dataset
from DRPU.drpu import DRPU_custom
from TCPU.tcpu import TCPU
from KM.Kernel_MPE import KM2_estimate


# Parameters:
c = 0.25 #label frequency
n_sample = 2000 #size of source/target data
pi_train = 0.1 #true source class prior
pi_test = 0.4 #true target class prior
sym = 1 #random seed 
sym_test = sym + 1000 #random seed

# Generate artiificial dataset:
Y_train, X_train = generate_artificial_data(20000,p=10,pi=pi_train,a=1,sym=sym)
Y_test, X_test = generate_artificial_data(n_sample,p=10,pi=pi_test,a=1,sym=sym_test)

# Generate PU source dataset:
X_train_pos, X_train_un = make_pu_cc_dataset(Y_train,X_train,pi_train=pi_train,n_sample=n_sample,c=c,sym=sym)

   
### Target class prior estimators: ###

# TCPU:
hat_pi_train, hat_pi_test = TCPU(X_train_pos,X_train_un,X_test,pi_train=None)
print("TPCU, estimator of target class prior = ",hat_pi_test)


# TCPU (known pi_train):
hat_pi_train, hat_pi_test = TCPU(X_train_pos,X_train_un,X_test,pi_train=pi_train)
print("TPCU (known pi_train), estimator of target class prior = ",hat_pi_test)

# DRPU:
X_pos = X_train_pos
X_un = X_train_un
acc, hat_pi_train, hat_pi_test = DRPU_custom(X_pos,X_un,X_test,Y_test,max_epochs = 200,batch_size = 100,lr = 2e-5,device_num=0,pi_train=None)
print("DRPU, estimator of target class prior = ",hat_pi_test)  


# DRPU (known pi_train):
X_pos = X_train_pos
X_un = X_train_un
acc, hat_pi_train, hat_pi_test = DRPU_custom(X_pos,X_un,X_test,Y_test,max_epochs = 200,batch_size = 100,lr = 2e-5,device_num=0,pi_train=pi_train)
print("DRPU (known pi_train), estimator of target class prior = ",hat_pi_test)  


# KM2-LS:
hat_pi_test = KM2_estimate(X_train_pos, X_test)
print("KM2-LS, estimator of target class prior = ",hat_pi_test)                  
                







