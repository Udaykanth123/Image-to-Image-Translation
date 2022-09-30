import torch
from torch import nn
import math


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        channels=64
        self.block1=nn.Sequential(
        nn.Conv2d(3,channels,kernel_size=9,padding=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels,channels*2,kernel_size=5,padding=2,stride=2),
        nn.InstanceNorm2d(channels*2),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels*2,channels*4,kernel_size=5,padding=2,stride=2),
        nn.InstanceNorm2d(channels*4),
        nn.ReLU(inplace=True),
        # nn.Conv2d(channels*4,channels*6,kernel_size=5,padding=2,stride=2),
        # nn.InstanceNorm2d(channels*6),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(channels*4,channels*6,kernel_size=5,padding=2,stride=2),
        # nn.InstanceNorm2d(channels*6),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(channels*6,channels*8,kernel_size=5,padding=2,stride=2),
        # nn.InstanceNorm2d(channels*8),
        # nn.ReLU(inplace=True)
        )

        block_res=[Residualblock(channels*4) for i in range(6)]
        self.block2=nn.Sequential(*block_res)
        
        self.block3=nn.Sequential(
        # nn.ConvTranspose2d(channels*8,channels*6,kernel_size=4,stride=2,padding=1),
        # nn.InstanceNorm2d(channels*6),
        # nn.ReLU(inplace=True),
        # nn.ConvTranspose2d(channels*6,channels*4,kernel_size=4,stride=2,padding=1),
        # nn.InstanceNorm2d(channels*4),
        # nn.ReLU(inplace=True),
        # nn.ConvTranspose2d(channels*6,channels*4,kernel_size=4,stride=2,padding=1),
        # nn.InstanceNorm2d(channels*4),
        # nn.ReLU(inplace=True),
        nn.ConvTranspose2d(channels*4,channels*2,kernel_size=4,stride=2,padding=1),
        nn.InstanceNorm2d(channels*2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(channels*2,channels,kernel_size=4,stride=2,padding=1),
        nn.InstanceNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels,3,kernel_size=9,padding=4)
        )

    def forward(self,x):
        r1=self.block1(x)
        r2=self.block2(r1)
        r3=self.block3(r2)

        return (torch.tanh(r3))



class Discriminator(nn.Module):
    def __init__(self):
        k=0.2
        super(Discriminator,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.LeakyReLU(k,inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(k,inplace=True),

            # nn.Conv2d(64,128,kernel_size=3,padding=1),
            # nn.InstanceNorm2d(128),
            # nn.LeakyReLU(k,inplace=True),

            # nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
            # nn.InstanceNorm2d(128),
            # nn.LeakyReLU(k,inplace=True),

            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(k,inplace=True),

            # nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
            # nn.InstanceNorm2d(256),
            # nn.LeakyReLU(k,inplace=True),

            # nn.Conv2d(256,512,kernel_size=3,padding=1),
            # nn.InstanceNorm2d(512),
            # nn.LeakyReLU(k,inplace=True),

            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(k,inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512,1024,kernel_size=1),
            nn.LeakyReLU(k,inplace=True),
            nn.Conv2d(1024,1,kernel_size=1)
        )

    def forward(self,x):
        batch_size = x.size(0)
        return torch.sigmoid(self.block(x).view(batch_size))



class Residualblock(nn.Module):
    def __init__(self,channels):
        super(Residualblock,self).__init__()
        
        self.block1=nn.Sequential(
            nn.Conv2d(channels,channels,3,stride=1,padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,3,stride=1,padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        r=self.block1(x)
        return r+x
    
    