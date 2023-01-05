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

torch.manual_seed(0)
torch.random.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

n_layers, n_filters, kernel_size = 10, 48, 3
channel_size = 3
orthogonal_init = False

#Using input convex neural network(ICNN) to denoise images
class ICNN(nn.Module):
    def __init__(self, n_in_channels=channel_size, n_filters=n_filters, kernel_size=kernel_size, n_layers=n_layers, orthogonal_init = False):
        super(ICNN, self).__init__()
        
        self.n_layers = n_layers
        self.orthogonal_init = orthogonal_init
        #these layers should have non-negative weights
        self.wz = nn.ModuleList([nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=1, bias=False)\
                                 for _ in range(self.n_layers)])
        
        #these layers can have arbitrary weights
        self.wx = nn.ModuleList([nn.Conv2d(n_in_channels, n_filters, kernel_size=kernel_size, stride=1, padding=1, bias=True)\
                                 for _ in range(self.n_layers+1)])
        
        #one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(n_filters, channel_size, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        
        #slope of leaky-relu
        self.negative_slope = 0.2 
        
    def forward(self, x):
        z = torch.nn.functional.leaky_relu(self.wx[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx[layer+1](x), negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        #z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)
        return z
    
    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        if self.orthogonal_init:
          random_init = torch.load('orthogonal_weight3.pt').to(device)
          for layer in range(self.n_layers):
            min_val + (max_val - min_val)\
            * torch.rand(n_filters, n_filters, kernel_size, kernel_size).to(device)
          self.final_conv2d.weight.data = min_val + (max_val - min_val)\
          * torch.rand(channel_size, n_filters, kernel_size, kernel_size).to(device)
          for layer in range(self.n_layers):
            for i in range(n_filters):
              for j in range(channel_size):
                self.wx[layer].weight.data[i,j,:,:] = random_init[i,j,:,:]

        else:
          for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val)\
            * torch.rand(n_filters, n_filters, kernel_size, kernel_size).to(device)

            self.wx[layer].weight.data = torch.randn(n_filters, channel_size, kernel_size, kernel_size).to(device)

          self.final_conv2d.weight.data = min_val + (max_val - min_val)\
          * torch.rand(channel_size, n_filters, kernel_size, kernel_size).to(device)
        print('weights initialized')
        return self
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.final_conv2d.weight.data.clamp_(0)
        return self 