# TCPU: Target Class prior estimator for Positive-Unlabled data under label shift

This is the repository for the paper 
>Prior shift estimation for positive unlabeled data through the lens of kernel embedding.

## Abstract ##

We study estimation  of a class prior  for unlabeled target samples which  possibly differs from that of source population. Moreover, it is assumed that the source data is partially observable: 
only samples from the positive class and from the whole population are available (PU learning scenario). We introduce a novel direct estimator of the class prior which avoids estimation of posterior probabilities in both populations and has a simple geometric interpretation. It is based on a distribution matching technique together with kernel embedding in Reproducing Kernel Hilbert  Space and is obtained  as an explicit solution to an optimisation task. We establish its asymptotic consistency as well as an explicit non-asymptotic    bound  on its deviation from the unknown prior, which is calculable in practice. We study  finite sample behaviour for synthetic and real data and show that the proposal works consistently on par or better than its competitors.

## Data ##

Pre-processed UCI data sets (Segment, Diabetes, Spambase, Waveform-5000, Banknote-authentication, Yeast and Vehicle), ready for analysis, are provided in ``data`` folder. 

The file ``prepare_image_data.py`` contains the code needed to download and prepare image datasets (MNIST, CIFAR and FAHION), as described in the paper.


## Experiments ##

1. The ``exp1_artificial.py`` ​​file contains the code to run experiments on artificial dataset.
2. The ``exp1_bench.py`` ​​file contains the code to run experiments on UCI data.
3. The ``exp1_images.py`` ​​file contains the code to run experiments on image datasets (CIFAR, MNIST, FASHION).
4. The ``exp2_artificial.py`` ​​file contains code for running experiments on artificial data in which we analyze the effect of sample size on the results.
5. ​The ​``exp3_artificial.py``file contains code for running experiments on artificial data in which we analyze the impact of source class prior estimation on the performance of the TCPU estimator.
6. The  ​``exp4_artificial.py``file contains code for running experiments on artificial data in which we analyze the robustness of TCPU to violation of the Label Shift (LS) assumption.

Visualizations:

1. The ``visualize_PU_label_shift.py`` file contains the code to generate  visualization of the label shift for PU data (Figure 1 in the main paper).
2. The ``visualize_kl_objective.py`` ​file contains the code to generate  visualization of the objective function behavior for different distribution shifts (Figure 2 in the main paper).



Example usage
--------
```python
from datasets import generate_artificial_data, make_pu_cc_dataset
from DRPU.drpu import DRPU_custom
from TCPU.tcpu import TCPU
from KM.Kernel_MPE import KM2_estimate


# Parameters:
c = 0.25 #label frequency
n_sample = 2000 #size of source/target data
pi_train = 0.1 #true source class prior
pi_test = 0.8 #true target class prior
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
```



