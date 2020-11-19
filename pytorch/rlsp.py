import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import params
from functions import shuffle_down, shuffle_up


# RLSP architecture with RGB output (Y output in the paper)
class RLSP(nn.Module):

    def __init__(self):
        super(RLSP, self).__init__()

        # define / retrieve model parameters
        factor = params["factor"]
        filters = params["filters"]
        kernel_size = params["kernel size"]
        layers = params["layers"]
        state_dim = params["state dimension"]
        device = params["device"]

        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(3*3 + 3*factor**2 + state_dim, filters, kernel_size, padding=int(kernel_size/2))
        self.conv_list = nn.ModuleList([nn.Conv2d(filters, filters, kernel_size, padding=int(kernel_size/2)) for _ in range(layers-2)])
        self.conv_out = nn.Conv2d(filters, 3*factor**2 + state_dim, kernel_size, padding=int(kernel_size/2))

    def cell(self, x, fb, state):

        # retrieve parameters
        factor = params["factor"]

        # define network
        res = x[:, 1]  # keep x for residual connection

        input = torch.cat([x[:, 0], x[:, 1], x[:, 2],
                           shuffle_down(fb, factor),
                           state], -3)

        # first convolution                   
        x = self.act(self.conv1(input))

        # main convolution block
        for layer in self.conv_list:
                x = self.act(layer(x))

        x = self.conv_out(x)
        
        out = shuffle_up(x[..., :3*factor**2, :, :] + res.repeat(1, factor**2, 1, 1), factor)
        state = self.act(x[..., 3*factor**2:, :, :])

        return out, state

    def forward(self, x):
        
        # retrieve device
        device = params["device"]

        # retrieve parameters
        factor = params["factor"]
        state_dimension = params["state dimension"]

        seq = []
        for i in range(x.shape[1]):                

            if i == 0:
                out = shuffle_up(torch.zeros_like(x[:, 0]).repeat(1, factor**2, 1, 1), factor)
                state = torch.zeros_like(x[:, 0, 0:1, ...]).repeat(1, state_dimension, 1, 1)

                out, state = self.cell(torch.cat([x[:, i:i+1], x[:, i:i+2]], 1).to(device), out, state)

            elif i == x.shape[1]-1:
                
                out, state = self.cell(torch.cat([x[:, i-1:i+1], x[:, i:i+1]], 1).to(device), out, state)

            else:
                
                out, state = self.cell(x[:, i-1:i+2], out, state)

            seq.append(out)

        seq = torch.stack(seq, 1)
        
        return seq
