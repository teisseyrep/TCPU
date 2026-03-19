import numpy as np
import matplotlib.pyplot as plt


# SOURCE DATA:
n=250
p=2
prior = 0.2
c = 0.4
y = np.zeros(n)
for i in np.arange(0,n,1):
    y[i] = np.random.binomial(1, prior, size=1)

X=np.zeros((n,p))
ind1 = np.where(y==1)[0]
ind0 = np.where(y==0)[0]
n1 = ind1.shape[0]
n0 = ind0.shape[0]

rho=0
Sigma1 = np.zeros((p,p))
Sigma0 = np.zeros((p,p))
for i1 in np.arange(p):
    for j1 in np.arange(p):
       Sigma1[i1,j1] = np.power(rho,np.abs(i1-j1))
       Sigma0[i1,j1] = np.power(rho,np.abs(i1-j1))

mean0 = [1,1]
mean1 = [5,5]

for j in np.arange(0,p,1):
        X[ind0,:]=np.random.multivariate_normal(mean0,Sigma0,size=n0)
        X[ind1,:]=np.random.multivariate_normal(mean1,Sigma1,size=n1)

ex_true = np.zeros(n)   
s = np.zeros(n)  
for i in np.arange(0,n,1):
        ex_true[i] = c

for i in np.arange(0,n,1):
    if y[i]==1:
        s[i]=np.random.binomial(1, ex_true[i], size=1)

w1 = np.where(y==1)[0]
w0 = np.where(y==0)[0]
ws1 = np.where(s==1)[0]

s1 = np.repeat(90,w0.shape[0])
s2 = np.repeat(90,w1.shape[0])
s3 = np.repeat(110,ws1.shape[0])

from scipy.stats import multivariate_normal
gsize = 100
v=np.linspace(np.min(X),np.max(X),gsize)
x,y = np.meshgrid(v,v)
norm1 = multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]])
norm2 = multivariate_normal(mean=[5,5], cov=[[1,0],[0,1]])
z = np.zeros((gsize,gsize))
for i in np.arange(gsize):
    for j in np.arange(gsize):
        point = [x[i,j],y[i,j]]
        z[i,j] = (1-prior) * norm1.pdf(point) + (prior)* norm2.pdf(point)



title1 = r'SOURCE PU data' + "\n" + r'$\pi=0.2$'
plt.style.use('default')
plt.rcParams.update({'font.size': 17})
plt.title(title1,fontsize=25)
plt.contourf(x, y, z, levels=500, cmap="Oranges")
plt.colorbar(format="%.2f")
plt.scatter(X[w0,0], X[w0,1], marker="_",color="black",label="negative (unlabeled)",s=s1)
plt.scatter(X[w1,0], X[w1,1], marker="+",color="black",label="positive (unlabeled)",s=s2)
plt.scatter(X[ws1,0], X[ws1,1], marker="+",color="blue",label="positive (labeled)",s=s3)
plt.legend(loc="lower right", ncol=1,prop = { "size": 14})
plt.axis("tight")
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)
name_out = 'visualize_LS_SOURCE.pdf' 
plt.tight_layout(pad=0.2)
plt.savefig(name_out, format="pdf")
plt.clf()



# TARGET DATA:
n=250
p=2
prior = 0.8
c=0.5
y = np.zeros(n)
for i in np.arange(0,n,1):
    y[i] = np.random.binomial(1, prior, size=1)

X=np.zeros((n,p))
ind1 = np.where(y==1)[0]
ind0 = np.where(y==0)[0]
n1 = ind1.shape[0]
n0 = ind0.shape[0]

rho=0
Sigma1 = np.zeros((p,p))
Sigma0 = np.zeros((p,p))
for i1 in np.arange(p):
    for j1 in np.arange(p):
       Sigma1[i1,j1] = np.power(rho,np.abs(i1-j1))
       Sigma0[i1,j1] = np.power(rho,np.abs(i1-j1))

mean0 = [1,1]
mean1 = [5,5]

for j in np.arange(0,p,1):
        X[ind0,:]=np.random.multivariate_normal(mean0,Sigma0,size=n0)
        X[ind1,:]=np.random.multivariate_normal(mean1,Sigma1,size=n1)



w1 = np.where(y==1)[0]
w0 = np.where(y==0)[0]

s1 = np.repeat(90,w0.shape[0])
s2 = np.repeat(90,w1.shape[0])

from scipy.stats import multivariate_normal
gsize = 100
v=np.linspace(np.min(X),np.max(X),gsize)
x,y = np.meshgrid(v,v)
norm1 = multivariate_normal(mean=[1,1], cov=[[1,0],[0,1]])
norm2 = multivariate_normal(mean=[5,5], cov=[[1,0],[0,1]])
z = np.zeros((gsize,gsize))
for i in np.arange(gsize):
    for j in np.arange(gsize):
        point = [x[i,j],y[i,j]]
        z[i,j] = (1-prior) * norm1.pdf(point) + (prior)* norm2.pdf(point)



title1 =  r'TARGET UNLABELED data' + "\n" +r'$\pi^{\prime}=0.8$'
plt.style.use('default')
plt.rcParams.update({'font.size': 17})
plt.title(title1,fontsize=25)
plt.contourf(x, y, z, levels=500, cmap="Oranges")
plt.colorbar(format="%.2f")
plt.scatter(X[w0,0], X[w0,1], marker="_",color="black",label="negative (unlabeled)",s=s1)
plt.scatter(X[w1,0], X[w1,1], marker="+",color="black",label="positive (unlabeled)",s=s2)
plt.legend(loc="lower right", ncol=1,prop = { "size": 14})
plt.axis("tight")  
ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)
name_out = 'visualize_LS_TARGET.pdf' 
plt.tight_layout(pad=0.2)
plt.savefig(name_out, format="pdf")
plt.clf()



