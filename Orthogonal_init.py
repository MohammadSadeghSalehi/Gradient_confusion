import torch
from torch import nn
kernel_size = 3
input_size = 32
number_of_filters = 48
class near_orthogonal_initialiser(torch.nn.Module):
    def __init__(self, n_in_channels=1, n_filters=1, kernel_size=kernel_size):
        super(near_orthogonal_initialiser, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size, stride = 1, padding = 2, bias= False)
        self.convT = nn.ConvTranspose2d(1, 1, kernel_size, stride = 1, padding = 2, bias = False)
    def forward(self, x):
        return (torch.nn.functional.leaky_relu(self.convT(torch.nn.functional.leaky_relu(self.conv(x), negative_slope=0.2)), negative_slope=0.2))
    def initialise(self):
        with torch.no_grad():
            self.conv.weight.data = torch.randn(1,1,kernel_size,kernel_size)
        with torch.no_grad():
            _ = self.convT.weight.copy_ (self.conv.weight)     
    def loss_function(self, x):
        return torch.linalg.norm(self.forward(x) - x)**2 

model = near_orthogonal_initialiser()
optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=500, max_eval=None, tolerance_grad=1e-08, tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")
def closure():
    optimizer.zero_grad()
    loss = model.loss_function(x)
    loss.backward()
    return loss

x = torch.randn((1,1,input_size,input_size))
def generate_orthogonal_weight(model,optimiser):
    model.initialise()
    optimizer.step(closure)
    return model.conv.weight.data
sample_size = 2000
sample_set = []    
weight = torch.empty(number_of_filters,number_of_filters,kernel_size,kernel_size)
for i in range(sample_size):
    sample = generate_orthogonal_weight(model,optimizer)
    while torch.isnan(sample).any() or model.loss_function(x) > 1e-6 or torch.linalg.norm(sample) > 1:
        sample = generate_orthogonal_weight(model,optimizer)
    sample_set.append(sample)
    print("Generated sample: ", i)
for i in range(number_of_filters):
    for j in range(number_of_filters):
        weight[i,j,:,:] = sample_set[torch.randint(0,sample_size,(1,))]

torch.save(weight, 'orthogonal_weight3.pt')
print(torch.isnan(weight).any())
