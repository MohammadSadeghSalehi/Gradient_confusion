# Reconstruction and data visualization
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
import os
from NN import ICNN
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Lambda

pic_size = 96
n_layers, n_filters, kernel_size = 16, 48, 3
channel_size = 3
num_epochs = 100

#load the data 
testset = torchvision.datasets.STL10(root = 'data' , split = 'test', download = False, transform = ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
# testset = torch.utils.data.Subset(testset, list(range(0,50)))
# testset = datasets.CIFAR10('test', download = True, train = False, transform = ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
testset = torch.utils.data.Subset(testset, list(range(640)))
testloader  = torch.utils.data.DataLoader(testset , batch_size = 640, shuffle = True)
test_iter = iter(testloader)
test_True,temp = next(test_iter)
noise = (torch.randn((640,channel_size,pic_size,pic_size),) * (0.25)**2).requires_grad_(False)
noisy_set = test_True + noise
model = ICNN(n_layers= 40)

#load the model
model.load_state_dict(torch.load('STL/ICNN_STLOrth40.pt'))
colorSet = ['red','cyan','olive','orange','black','brown','purple','yellow','pink','gray','green','blue']
lineType = ['solid','dashed','dotted','dashdot']
    
NewAverage = torch.zeros((num_epochs))
index = [16,24,32,40]
for i in range(len(index)):
    Run_num = f'ICNN_CifarRand{str(index[i])}'
    train_loss_avg, test_loss_avg = torch.load(f'{os.getcwd()}/ICNN_Cifar/Orth/{Run_num}_train_loss_avg.pt'), torch.load(f'{os.getcwd()}/ICNN_Cifar/Orth/{Run_num}_test_loss_avg.pt')
    for j in range(num_epochs):
        NewAverage[j] = torch.mean(train_loss_avg[:j])
    plt.plot(np.linspace(7,num_epochs,num_epochs-7),(NewAverage[7:]).cpu().detach().numpy(),color=colorSet[i],linestyle=lineType[int(torch.randint(0,3,(1,)).detach().numpy())],linewidth=1.5,label= f'Train{str(index[i])}layers')
    # plt.plot(np.linspace(1,num_epochs,num_epochs),(test_loss_avg).cpu().detach().numpy(),color=colorSet[i+2],linestyle='dashed',linewidth=2,label=f'Test{str(2**i)}layers')
plt.legend()
plt.title('Denoising Using ICNN with inexact Orthogonal Initialisation')
plt.xlabel('Epochs')
plt.ylabel('Average MSE Loss')
# plt.xscale('log')
plt.yscale('log')
plt.savefig('Train_Orth_ICNN.png',dpi=300)
plt.show()

#Plot the denoised images
denoise_list = []
noisy_list = []
psnr_noisy_list = []
psnr_denoised_list = []
index = [10,55,104,210, 263 ,398]
def psnr(x, y):
    return 10*torch.log10(pic_size*pic_size/(torch.linalg.norm(x-y)**2))
import itertools
for i in index:
    denoise_list.append(model(noisy_set[i,:,:,:].unsqueeze(0)).detach().cpu())
    psnr_denoised_list.append(psnr(denoise_list[-1],test_True[i,:,:,:].unsqueeze(0)))
    noisy_list.append(noisy_set[i,:,:,:].detach().cpu().unsqueeze(0))
    psnr_noisy_list.append(psnr(noisy_list[-1],test_True[i,:,:,:].unsqueeze(0)))
# define subplot grid
fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(7, 3))
plt.subplots_adjust(hspace=0.05, wspace=0.5)
fig.suptitle("STL-10 test, ICNN-40 with Orthogonal initialisation", fontsize=12, y=0.99)
for i, j in itertools.product(range(2), range(6)):
    axs[0,j].imshow(denoise_list[j].squeeze().permute(1,2,0))
    axs[0,j].set_title(f'PSNR: {psnr_denoised_list[j].item():.2f}',fontsize=8)
    axs[1,j].imshow(noisy_list[j].squeeze().permute(1,2,0))
    axs[1,j].set_title(f'PSNR: {psnr_noisy_list[j].item():.2f}',fontsize=8)
    # axs[i,j].set_xlabel(f'Index: {index[i*2+j]}')
    axs[i,j].set_xticks([])
    axs[i,j].set_yticks([])
plt.savefig('Denoised_Orth_ICNN_3.png',dpi=300)
plt.show()



# High resolution denoising
high = plt.imread('/Users/sadegh/Desktop/Htest.jpeg')
plt.imshow(high)
plt.show()
pic_size = 512
noise = (torch.randn((1,channel_size,512,512),) * (0.25)**2).requires_grad_(False)
high = (torch.tensor(high)/255).permute(2,0,1).unsqueeze(0)
denoised = model(high+noise)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
fig.suptitle("High resolution generalisation, ICNN-40 with Orthogonal initialisation", fontsize=12, y=0.99)
plt.subplots_adjust(hspace=0.05, wspace=0.05)
axs[0].imshow(denoised.squeeze().permute(1,2,0).detach().numpy())
axs[0].set_title(f'PSNR: {psnr(denoised.squeeze(),high.squeeze()):.2f}',fontsize=8)
axs[1].imshow((high+noise.cpu()).squeeze().permute(1,2,0).detach().numpy())
axs[1].set_title(f'PSNR: {psnr(high.squeeze(),(high+noise.cpu()).squeeze()):.2f}',fontsize=8)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_xticks([])
axs[1].set_yticks([])
plt.savefig('denoised_high_res_STL_Orth_1.png',dpi=300)
plt.show()
