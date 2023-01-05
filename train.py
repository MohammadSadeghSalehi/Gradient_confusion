import numpy as np
import torch
import torchvision
import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
import os
import gc
from NN import ICNN
torch.manual_seed(0)
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layers, n_filters, kernel_size = 10, 48, 3
channel_size = 3
batch_size = 32
pic_size = 96
def loss(x,y,model):
    return torch.linalg.norm(model(x)-y)**2

num_epochs = 50
for k in [40]:
    n_layers = k
    Run_num = f'ICNN_STLRand{str(k)}'
    # Load the data
    from torchvision.transforms import ToTensor, Lambda
    # trainset    = datasets.MNIST('train', download = True, train = True, transform=ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    # trainset = datasets.CIFAR10('train', download = True, train = True, transform=ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    trainset = torchvision.datasets.STL10(root = 'data' , split = 'train', download = True, transform = ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    trainset = torch.utils.data.Subset(trainset, list(range(3200)))
    # valset      = datasets.MNIST('val  ', download = True, train = False,transform = None)
    #valset      = datasets.FashionMNIST('val', download = True, train = False, transform = transform)
    # valset = torch.utils.data.Subset(valset, list(range(500)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size =batch_size, shuffle = True)
    # valloader   = torch.utils.data.DataLoader(valset  , batch_size = 5, shuffle = True)
    # testset     = datasets.MNIST('test ', download = True, train = False,transform=ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    testset = torchvision.datasets.STL10(root = 'data' , split = 'test', download = True, transform = ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    # testset = datasets.CIFAR10('test', download = True, train = False, transform = ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    testset = torch.utils.data.Subset(testset, list(range(640)))
    testloader  = torch.utils.data.DataLoader(testset , batch_size = batch_size, shuffle = True)
    dataiter = iter(trainloader)
    test_iter = iter(testloader)
    test_final,temp = next(test_iter)
    torch.cuda.empty_cache()
    model = ICNN(n_layers=n_layers,orthogonal_init = False)
    print(model)
    model.initialize_weights()
    #model.load_state_dict(torch.load(f'{os.getcwd()}/ICNN.pt'))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999), eps=1e-09, weight_decay=1, amsgrad=True)
    train_loss = []
    test_loss = []
    train_loss_avg = []
    test_loss_avg = []
    X_test = torch.empty(batch_size,channel_size,pic_size,pic_size,device = device,requires_grad= False)
    y_test = torch.empty(batch_size,channel_size,pic_size,pic_size)
    denoised = torch.empty(batch_size,channel_size,pic_size,pic_size,device = device,requires_grad= False)
    noise = (torch.randn((3200,channel_size,pic_size,pic_size),) * (0.25)**2).to(device).requires_grad_(False)
    batch_size = 32
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            y = data[0].to(device).requires_grad_(False)
            X = (y+noise[i*y.shape[0]:(i+1)*y.shape[0],:,:,:]).to(device).requires_grad_(False)
            model.to(device)
            optimiser.zero_grad()
            loss_val = loss(X,y,model)
            loss_val.backward()
            optimiser.step()
            train_loss.append(loss_val.cpu() / (pic_size**2*batch_size))
            data[0].cpu()
            y.cpu()
            X.cpu()
            loss_val.cpu()

        torch.cuda.empty_cache()
        gc.collect()
        for i, data in enumerate(testloader, 0):
            X_test = (data[0].to(device).requires_grad_(False)+noise[i*data[0].shape[0]:(i+1)*data[0].shape[0],:,:,:])
            model.eval()
            with torch.no_grad():
                denoised = model(X_test)
            test_loss.append(
                (
                    torch.linalg.norm((X_test) - (denoised)) ** 2
                    / (pic_size**2*batch_size)
                ).cpu()
            )
            y.cpu()
            X.cpu()
            data[0].cpu()
            gc.collect()
        torch.save(model.cpu().state_dict(), f'{os.getcwd()}/{Run_num}.pt')
        if epoch%5 == 0:
            optimiser.param_groups[0]['lr'] = optimiser.param_groups[0]['lr']*(0.9)
        torch.cuda.empty_cache()
        gc.collect()
        train_loss_avg.append(torch.mean(torch.stack(train_loss)[epoch*len(trainloader):(epoch+1)*len(trainloader)]))
        test_loss_avg.append(torch.mean(torch.stack(test_loss)[epoch*len(testloader):(epoch+1)*len(testloader)]))
        print("Epoch ",epoch+1," Train MSE: ", train_loss_avg[-1])
        print("Epoch ",epoch+1," Test MSE: ", test_loss_avg[-1])

    torch.save(
        torch.stack(train_loss_avg).cpu().detach(),
        f'{os.getcwd()}/{Run_num}_train_loss_avg.pt',
    )
    torch.save(
        torch.stack(test_loss_avg).cpu().detach(),
        f'{os.getcwd()}/{Run_num}_test_loss_avg.pt',
    )
    torch.save(
        torch.stack(train_loss).cpu().detach(),
        f'{os.getcwd()}/{Run_num}_train_loss.pt',
    )
    torch.save(
        torch.stack(test_loss).cpu().detach(),
        f'{os.getcwd()}/{Run_num}_test_loss.pt',
    )


