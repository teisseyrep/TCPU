import numpy as np
from sklearn.model_selection import train_test_split
from DRPU.model import LinearClassifier, MultiLayerPerceptron
from DRPU.metric import *
from DRPU.algorithm import *
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from KM.Kernel_MPE import KM2_estimate

def DRPU_custom(X_pos,X_un,X_test,Y_test,max_epochs = 1000,batch_size = 100,lr = 2e-5,device_num=0,pi_train=None,alpha=None,model="MLP"):

    loss_name = "LSIF"

    Y_pos = np.ones(X_pos.shape[0])
    Y_un = -np.ones(X_un.shape[0])
    X_train_pos, X_val_pos, Y_train_pos, Y_val_pos = train_test_split(X_pos, Y_pos, test_size=0.5, random_state=42)
    X_train_un, X_val_un, Y_train_un, Y_val_un = train_test_split(X_un, Y_un, test_size=0.5, random_state=42)

    trainTensor_P = TensorDataset(torch.tensor(X_train_pos, dtype=torch.float32), torch.tensor(Y_train_pos, dtype=torch.float32))
    trainloader_P = DataLoader(dataset=trainTensor_P, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    valTensor_P = TensorDataset(torch.tensor(X_val_pos, dtype=torch.float32), torch.tensor(Y_val_pos, dtype=torch.float32))
    valloader_P = DataLoader(dataset=valTensor_P, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    Y_train_un = -np.ones(X_train_un.shape[0])
    Y_val_un = -np.ones(X_val_un.shape[0])

    trainTensor_U = TensorDataset(torch.tensor(X_train_un, dtype=torch.float32), torch.tensor(Y_train_un, dtype=torch.float32))
    trainloader_U = DataLoader(dataset=trainTensor_U, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    valTensor_U = TensorDataset(torch.tensor(X_val_un, dtype=torch.float32), torch.tensor(Y_val_un, dtype=torch.float32))
    valloader_U = DataLoader(dataset=valTensor_U, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    testTensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
    testloader = DataLoader(dataset=testTensor, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)


    device = torch.device(device_num) if device_num >= 0 and torch.cuda.is_available() else 'cpu'
    
    if(model=="MLP"):
        model = MultiLayerPerceptron(dim=X_train_pos.shape[1])    
    elif(model=="LC"):
        means  = np.mean(X_train_un,1).reshape(-1,1)
        means = torch.tensor(means, dtype=torch.float32)
        model = LinearClassifier(means, activate_output=True)
    

    #if pi_train is not None:
    #    alpha = pi_train
    #elif alpha is None:
    #    alpha = 0

    if pi_train is not None:
        alpha = pi_train
    else:
        if alpha is None:
            hat_pi_train = KM2_estimate(X_pos,X_un)    
            alpha = hat_pi_train

    #print("alpha=",alpha)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.1, betas=(0.5, 0.999))
    #criterion = BregmanDivergence(choose_loss(loss_name))
    criterion = NonNegativeBregmanDivergence(alpha, choose_loss(loss_name))
    criterion_val = BregmanDivergence(choose_loss(loss_name))


    train_loss_vec=[]
    validation_loss_vec=[]

    for ep in range(max_epochs):
        # train step
        #print(ep)
        model.train()
        train_loss = []
        for (x_p, t_p), (x_u, t_u) in zip(trainloader_P, trainloader_U):
            
            num_P, num_U = len(x_p), len(x_u)
            x = torch.cat([x_p, x_u]).to(device)
            y = model(x).view(-1)
            y_p, y_u = y[:num_P], y[num_P:]
            loss = criterion(y_p, y_u)
            loss.backward()
            optimizer.step()
            train_loss.append(criterion.value())
        
        train_loss_vec.append(np.array(train_loss).mean())
        #print(np.array(train_loss).mean())

        # validation step
        model.eval()
        with torch.no_grad():
            validation_loss = []
            for (x_p, t_p), (x_u, t_u) in zip(valloader_P, valloader_U):
                num_P, num_U = len(x_p), len(x_u)
                x = torch.cat([x_p, x_u]).to(device)
                y = model(x).view(-1)
                y_p, y_u = y[:num_P], y[num_P:]
                criterion_val(y_p, y_u)
                validation_loss.append(criterion_val.value())
            validation_loss_vec.append(np.array(validation_loss).mean())
       
        
    # test step
    #train_prior, preds_P = estimate_train_prior(model, valloader_P, valloader_U, device)
    
    model.eval()
    with torch.no_grad():
        preds_P, preds_U = [], []
        for x_p, t_p in valloader_P:
            x = x_p.to(device)
            y = model(x).view(-1)
            preds_P.append(to_ndarray(y))
        for x_u, t_u in valloader_U:
            x = x_u.to(device)
            y = model(x).view(-1)
            preds_U.append(to_ndarray(y))
        preds_P = np.concatenate(preds_P)
        preds_U = np.concatenate(preds_U)
        
        if pi_train==None:
            train_prior = priorestimator(np.concatenate([preds_P, preds_U]), PUsequence(len(preds_P), len(preds_U)))
            test_prior = estimate_test_prior(model, testloader, preds_P, device)
        else:
            train_prior = pi_train
            
    test_prior = estimate_test_prior(model, testloader, preds_P, device)
    
    #if test_prior==0:
    #    test_prior = 0.5
    
    thresh = train_prior * (1 - test_prior) / (train_prior * ((1 - train_prior) * test_prior + train_prior * (1 - test_prior)) + EPS)
    acc, auc = prediction(model, testloader, device, thresh)

    return acc, train_prior,test_prior
