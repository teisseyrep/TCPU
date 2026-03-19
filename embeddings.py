import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable


def make_embedding_resnet18(trainset_data,ds='CIFAR10'):
   
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    layer = model._modules.get('avgpool')
    model.eval()
    if ds=='CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])
    elif ds=='MNIST':
        normalize =transforms.Normalize((0.1307,), (0.3081,))
    elif ds=='Fashion':
        normalize = transforms.Normalize((0.2860,), (0.3530,))
    else:
        raise Exception("ds not found!")
        
    to_tensor = transforms.ToTensor()
   
    n = len(trainset_data)
    X = np.zeros((n,512))
    for i in np.arange(0,n):
        if i % 1000==0 and i>0:
            print(i)
        img = trainset_data[i]   
        t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))
        my_embedding = torch.zeros(512)
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
        h = layer.register_forward_hook(copy_data)
        model(t_img)
        h.remove()
        
        X[i,:]=my_embedding.numpy()
    
    return(X)
