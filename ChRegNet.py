'''
Reference:
Arta Mohammad-Alikhani, Ehsan Jamshidpour, Sumedh Dhale, Milad Akrami, Subarni Pardhan, Babak Nahid-Mobarakeh, 
"Fault Diagnosis of Electric Motors by a Channel-wise Regulated CNN and Differential of STFT," IEEE Transactions on Industry Application, 2024.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
import typing


class RegulatorBlock(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=1):
        super().__init__()
        self.intermediate_channels = intermediate_channels
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels *  4,
            kernel_size=kernel_size, padding= (kernel_size // 2)*2,dilation=2, bias=True,
        )
        self.bn = nn.BatchNorm2d(intermediate_channels)
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> typing.Tuple:
        c, h = state
        h = h.to(device=x.device)
        c = c.to(device=x.device) 
        x = torch.cat([x, h], dim=1)
        x = self.conv_x(x)
        
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.tanh(b)
        g = torch.sigmoid(g)
        d = torch.tanh(d)
        c =  a * b * c + g * d
        return c

    def init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> typing.Tuple:
        length = spatial_dim
        c = torch.zeros(batch_size, channels, length, device=self.conv_x.weight.device)
        h = torch.zeros(batch_size, channels, length, device=self.conv_x.weight.device)
        return c, h
class Regulator_init(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=1):
        super().__init__()
        self.intermediate_channels = intermediate_channels
        self.conv_x = nn.Conv2d(
            x_in, intermediate_channels *  4,
            kernel_size=kernel_size, padding= (kernel_size // 2)*2,dilation=2, bias=True,
        )
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> typing.Tuple: 
        x = self.conv_x(x)
        
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        g = torch.sigmoid(g)
        d = torch.tanh(d)
        c =  a *  g * d
        h = b * torch.tanh(c)
        return c, h


class RegularBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, in_channels, n_class, use_feature=False):
        super().__init__()
        self.name = 'ChRegNet'
        self.use_feature = use_feature
        self.bn = nn.BatchNorm2d(in_channels)
        self.b0=nn.Conv2d(3, 3, kernel_size=(1,5), stride=(1,2))
        self.b1 = RegularBlock(in_channels, 32, kernel_size=(3,7), stride=(1,2))
        self.reg_init_cell = Regulator_init(16, 16, 3)
        self.b2 = RegularBlock(16, 16)
        self.b3 = RegularBlock(16, 32)
        self.b4 = RegularBlock(32, 32)
        self.reg_cell = RegulatorBlock(32, 16, 3)
        self.b5 = RegularBlock(16, 16)
        self.fc= nn.LazyLinear(n_class)
        self.pool = nn.MaxPool2d(1,2)
        self.flatten = nn.Flatten()
        
        

    def forward(self, x):
        f0 = self.bn(x)
        f0=torch.cat((f0[:,0:1,:,:]-f0[:,1:2,:,:],f0[:,0:1,:,:]-f0[:,2:3,:,:],f0[:,1:2,:,:]-f0[:,2:3,:,:]),1)
        f1 = self.b1(f0)
        B, Ch, L, W = f1.shape
        h = torch.zeros(B, Ch, L, W)
        c = torch.zeros(B, Ch, L, W)
        c, h = self.reg_init_cell(f1[:,:16,:,:], (c, h))
        f2 = self.b2(f1[:,16:,:,:])
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        c = self.reg_cell(f4, (c, h))
        f5 = self.b5(c)
        f5 = self.pool(f5)
        f6=self.flatten(f5)
        out=self.fc(f6)
        return out
    

