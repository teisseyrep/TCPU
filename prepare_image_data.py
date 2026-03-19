import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from embeddings import make_embedding_resnet18
from datasets import label_transform_MNIST, label_transform_CIFAR10 , label_transform_Fashion


ds = "MNIST"
   
   
if ds == 'MNIST':
    transformer = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    
    trainset = MNIST(root='./data', train=True, download=True, transform=transformer)
    testset = MNIST(root='./data', train=False, download=True, transform=transformer)         
    trainset.targets = label_transform_MNIST(trainset.targets)
    testset.targets = label_transform_MNIST(testset.targets)
    trainset.data = np.stack((trainset.data,)*3, axis=-1)
    testset.data = np.stack((testset.data,)*3, axis=-1) 
    batchsize = len(trainset)   
elif ds == 'CIFAR10':
    transformer = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transformer)
    trainset.targets = label_transform_CIFAR10(trainset.targets)
    testset.targets = label_transform_CIFAR10(testset.targets)
    trainset.targets = torch.tensor(trainset.targets)
    testset.targets = torch.tensor(testset.targets)
    

elif ds == 'Fashion':
    transformer = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))])
    
    trainset = FashionMNIST(root='./data', train=True, download=True, transform=transformer)
    testset = FashionMNIST(root='./data', train=False, download=True, transform=transformer) 
    trainset.targets = label_transform_Fashion(trainset.targets)
    testset.targets = label_transform_Fashion(testset.targets)   
    trainset.data = np.stack((trainset.data,)*3, axis=-1)
    testset.data = np.stack((testset.data,)*3, axis=-1)

    
subset1 = torch.arange(0,len(trainset.data))
subset2 = torch.arange(0,len(testset.data))


trainset_data = trainset.data[subset1,:,:]
testset_data = testset.data[subset2,:,:]
y = trainset.targets[subset1].numpy()
ytest = testset.targets[subset2].numpy()
X = make_embedding_resnet18(trainset_data,ds=ds)
Xtest = make_embedding_resnet18(testset_data,ds=ds)


Xy=np.append(X,y[:,None],1)
df_train=pd.DataFrame.from_records(Xy)

Xy_test=np.append(Xtest,ytest[:,None],1)
df_test=pd.DataFrame.from_records(Xy_test)


file_out = "data/" + ds + "_train" + ".csv" 
df_train.to_csv(file_out)
print(df_train) 

file_out = "data/" + ds + "_test" + ".csv" 
df_test.to_csv(file_out)
print(df_test) 

