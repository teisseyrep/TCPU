import numpy as np

from datasets import generate_artificial_data,  make_pu_cc_dataset
from TCPU.tcpu import TCPU
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt

c1 = 0.5
n_sample = 1000
n_sym = 20
p = 10
setting='1'


if setting=='1':
    pi_train = 0.2
    pi_test = 0.2
    correct = 0.005
if setting=='2':
    pi_train = 0.2
    pi_test = 0.5
    correct = 0 
if setting=='3':
    pi_train = 0.2
    pi_test = 0.8
    correct = 0 

res = np.zeros((100,20))
hat_pi_test_seq = np.zeros(20)

for sym in np.arange(n_sym):
 
    print(sym) 

    Y_train, X_train = generate_artificial_data(20000,p,pi=pi_train,a=3,sym=sym)
    X_train_pos, X_train_un = make_pu_cc_dataset(Y_train,X_train,pi_train=pi_train,n_sample=n_sample,c=c1,sym=sym)
    n_un = X_train_un.shape[0]
    Y_test, X_test = generate_artificial_data(1000,p,pi=pi_test,a=3,sym=sym)
    
    
    # TCPU (known pi_train):
    hat_pi_train, hat_pi_test = TCPU(X_train_pos,X_train_un,X_test,pi_train=pi_train)
    print("PULS (known pi_train)", hat_pi_test)
    hat_pi_test_seq[sym] = hat_pi_test
    
    gamma= None
    D  = np.sum( pairwise_kernels(X_train_un, metric='rbf',gamma=gamma) )
    E  = np.sum( pairwise_kernels(X_train_pos, metric='rbf',gamma=gamma) ) 
    K  = np.sum( pairwise_kernels(X_train_un, X_train_pos, metric='rbf',gamma=gamma) )
    L  = np.sum( pairwise_kernels(X_train_un, X_test, metric='rbf',gamma=gamma) )
    M  = np.sum( pairwise_kernels(X_train_pos, X_test, metric='rbf',gamma=gamma) )
    
    n = X_train_un.shape[0]
    m = X_train_pos.shape[0]
    nt = X_test.shape[0] 
    
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
    
    
    g_seq = np.linspace(start=0, stop=1, num=100)
    
    
    for i in np.arange(100):
        g  = g_seq[i]
        res[i,sym] = g**(2) * den - 2* g * num
    
    if np.min(res[:,sym])<0:
        res[:,sym] = res[:,sym] + np.abs(np.min(res[:,sym]))
    else:
        res[:,sym] = res[:,sym] - np.abs(np.min(res[:,sym]))




name_out = 'visualize_kl_objective' + str(pi_train) + str(pi_test) + '.pdf'

label1 = r'avg. $\widehat{\pi}^{\prime}$'
label2 = r'true $\pi^{\prime}$'
label2a = r'true $\pi$'
label3 = r'$\hat{\mathcal{L}}(\gamma)$'

if setting=='1':
    title1 = r'$\bf{No}$' + r' $\bf{shift}$' '\n' + r'$\pi=$'+'{}'.format(pi_train) + r' --> $\pi^{\prime}=$'+'{}'.format(pi_test)
if setting=='2':
    title1= r'$\bf{Moderate}$' + r' $\bf{shift}$' '\n' + r'$\pi=$'+'{}'.format(pi_train) + r' --> $\pi^{\prime}=$'+'{}'.format(pi_test)
if setting=='3':
    title1= r'$\bf{Large}$' + r' $\bf{shift}$' '\n' + r'$\pi=$'+'{}'.format(pi_train) + r' --> $\pi^{\prime}=$'+'{}'.format(pi_test)


plt.style.use('default')
plt.rcParams.update({'font.size': 17})
plt.plot(g_seq, res[:,0], color="g",label=label3,linewidth=1)
plt.plot(g_seq, res[:,1], color="g",linewidth=1)
plt.plot(g_seq, res[:,3], color="g",linewidth=1)
plt.plot(g_seq, res[:,4], color="g",linewidth=1)
plt.plot(g_seq, res[:,5], color="g",linewidth=1)
plt.plot(g_seq, res[:,6], color="g",linewidth=1)
plt.plot(g_seq, res[:,7], color="g",linewidth=1)
plt.plot(g_seq, res[:,8], color="g",linewidth=1)
plt.plot(g_seq, res[:,9], color="g",linewidth=1)
plt.plot(g_seq, res[:,10], color="g",linewidth=1)
plt.plot(g_seq, res[:,11], color="g",linewidth=1)
plt.plot(g_seq, res[:,12], color="g",linewidth=1)
plt.plot(g_seq, res[:,13], color="g",linewidth=1)
plt.plot(g_seq, res[:,14], color="g",linewidth=1)
plt.plot(g_seq, res[:,15], color="g",linewidth=1)
plt.plot(g_seq, res[:,16], color="g",linewidth=1)
plt.plot(g_seq, res[:,17], color="g",linewidth=1)
plt.plot(g_seq, res[:,18], color="g",linewidth=1)
plt.plot(g_seq, res[:,19], color="g",linewidth=1)
plt.vlines(np.mean(hat_pi_test_seq), 0, 0.2,color="blue",linewidth=3,label=label1)
plt.vlines(pi_test-correct, 0, 0.2,color="red",linestyle="dashed",linewidth=3,label=label2)
plt.vlines(pi_train+correct, 0, 0.2,color="m",linestyle="dashed",linewidth=3,label=label2a)
plt.xlabel(r'$\gamma$')
plt.ylabel("")
plt.title(title1,fontsize=30)
plt.legend(loc="lower center", ncol=2,prop = { "size": 20 },bbox_to_anchor=(0.5, -0.5))
plt.ylim((0,0.1))
plt.grid(which="both")
plt.axvspan(np.min(hat_pi_test_seq), np.max(hat_pi_test_seq), color='grey', alpha=0.2)
plt.xlim(0,1)
#plt.show()
plt.savefig(name_out, format="pdf", bbox_inches='tight')
plt.clf()

