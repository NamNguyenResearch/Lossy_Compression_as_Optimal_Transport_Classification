import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torch
from torch.autograd import Variable


# =============================================================================
# Define CNN Classier Model for MNIST dataset
class ClassifierMNIST(nn.Module):
    def __init__(self):
        super(ClassifierMNIST, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer
        self.fc1 = nn.Linear(4 * 4 * 10, 100)
        # Output layer
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (N, 10, 24, 24)
        x = self.pool(x)           # (N, 10, 12, 12)
        x = F.relu(self.conv2(x))  # (N, 10, 8, 8)
        x = self.pool(x)           # (N, 10, 4, 4)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# =============================================================================


# =============================================================================
# Define a ResNet-20 Model (for CIFAR/SVHN)
# =========================
# Define the basic residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        A basic residual block with two 3x3 convolutions.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet architecture
class ClassifierSVHN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        """
        ResNet for 32x32 images. The architecture is similar to that used for CIFAR-10.
        """
        super(ClassifierSVHN, self).__init__()
        self.in_planes = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three groups of residual blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
def ResNet20():
    """
    Constructs a ResNet-20 model for small images.
    """
    return ClassifierSVHN(BasicBlock, [3, 3, 3])
# =============================================================================


class Discriminator(nn.Module):
    def __init__(self, input_channel = 1):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(4096,1)

    def forward(self, x):
        x = self.main(x)
        x =x.view(-1,4096)
        x = self.fc(x)
        return x
    

def uniform_noise(size, alpha: float):
    # Want this to be in [-alpha/2, +alpha/2)
    return torch.rand(*size)*alpha - alpha/2

def generate_centers(L: int, limits: Tuple[float, float]):
    # Uniformly distributed between [limits[0], limits[1]]
    lower, upper = limits[0], limits[1]
    assert lower < upper
    interval = upper - lower
    centers = [lower + l/(L-1)*interval for l in range(0, L)]

    return centers

class Quantizer(nn.Module):
    """
    Scalar Quantizer module
    Source: https://github.com/mitscha/dplc
    """
    def __init__(self, centers=[-1.0, 1.0], sigma=1.0):
        super(Quantizer, self).__init__()
        self.centers = centers
        self.sigma = sigma

    def forward(self, x):
        centers = x.data.new(self.centers)
        xsize = list(x.size())

        # Compute differentiable soft quantized version
        x = x.view(*(xsize + [1]))
        level_var = Variable(centers, requires_grad=False)
        dist = torch.abs(x-level_var)
        output = torch.sum(level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)

        # Compute hard quantization (invisible to autograd)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)): centers.unsqueeze(0) # in-place error
        centers = centers.expand(*(xsize + [len(self.centers)]))

        quant = centers.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Generator_svhn(nn.Module):
    """Generator for denoising on SVHN"""
    def __init__(self, latent_dim, L, ql, stochastic, common):
        super(Generator_svhn, self).__init__()
        # encoding blocks
        self.n_channel = 3
        self.latent_dim = latent_dim
        self.ql = ql
        self.stoch = stochastic
        self.ls = 1
        self.in_size = 32
        self.L = L
        self.q = [-1,1]
        self.common = common
        self.output_size = 3*32*32
        self.encoder = Encoder_svhn(n_channel=self.n_channel, latent_dim=self.latent_dim, \
            quantize_latents = self.ql, stochastic=self.stoch,\
                 ls=self.ls, input_size=self.in_size, L=self.L, q_limits=self.q)
        
        # decoding blocks
        self.decoder = Decoder_svhn(latent_dim=self.latent_dim, output_size=self.output_size, stochastic = self.stoch)
        
    def forward(self, x):
        if not self.common:
            u1 = uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
            u2 = uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
        else:
            u1 = uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
            u2 = u1
        out = self.encoder(x,u1)
        out = self.decoder(out,u2)
        return out

class Encoder_svhn(nn.Module):
    def __init__(self, n_channel, latent_dim, quantize_latents, stochastic,
                 ls, input_size, L, q_limits):
        super(Encoder_svhn, self).__init__()

        self.n_channel = n_channel
        self.latent_dim = latent_dim
        self.quantize_latents = quantize_latents
        self.stochastic = stochastic
        self.ls = ls # layer scale: integer factor
        self.input_size = input_size # specified by dataset

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers, sigma=2/L)
        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)
            else:
                raise ValueError('Quant. disabled')

        ilw = int(self.ls*64) # initial layer width
        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(ilw, 2*ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(2*ilw, 4*ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.conv_flat_dim = ilw*4*4*4
        self.final = nn.Sequential(
            nn.Linear(self.conv_flat_dim, self.latent_dim),
            nn.Tanh(),
        )

    def forward(self, x, u):
        x = self.main(x)
        x = x.view(-1, self.conv_flat_dim)
        x = self.final(x)

        if self.stochastic:
            x = x + u

        if self.quantize_latents:
            x = self.q(x)

        return x
    
class Decoder_svhn(nn.Module):
    def __init__(self, latent_dim, output_size, stochastic):
        super(Decoder_svhn, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.stochastic = stochastic

        self.expand = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        if self.output_size == 3*32*32:
            self.deconvolve = nn.Sequential(
                nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 108, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(108),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(108, 128, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 3, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f'No deconvolution defined for output size of {self.output_size}.')

    def forward(self, x, u):
        if self.stochastic:
            x = x-u
        x = self.expand(x)
        x = x.view(-1, 32, 4, 4)
        x = self.deconvolve(x)

        return x


class Generator_mnist(nn.Module):
    """Generator for super-resolution on MNIST"""
    def __init__(self, latent_dim, L, ql, stochastic, common):
        super(Generator_mnist, self).__init__()
        # encoding blocks
        self.n_channel = 1
        self.latent_dim = latent_dim
        self.ql = ql
        self.stoch = stochastic
        self.ls = 1
        self.in_size = 32
        self.L = L
        self.q = [-1,1]
        self.common = common
        self.output_size = 784
        self.encoder = Encoder_mnist(n_channel=self.n_channel, latent_dim=self.latent_dim, \
            quantize_latents = self.ql, stochastic=self.stoch,\
                 ls=self.ls, input_size=self.in_size, L=self.L, q_limits=self.q)
        
        # decoding blocks
        self.decoder = Decoder_mnist(latent_dim=self.latent_dim, output_size=self.output_size, stochastic = self.stoch)
        
    def forward(self, x):
        if not self.common:
            u1 = uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
            u2 = uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
        else:
            # print('using common randomness!!!')
            u1 = uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
            u2 = u1#uniform_noise([x.size(0), self.latent_dim], self.encoder.alpha).cuda()
        out = self.encoder(x,u1)
        out = self.decoder(out,u2)
        return out

class Encoder_mnist(nn.Module):
    def __init__(self, n_channel, latent_dim, quantize_latents, stochastic,
                 ls, input_size, L, q_limits):
        super(Encoder_mnist, self).__init__()

        self.n_channel = n_channel
        self.latent_dim = latent_dim
        self.quantize_latents = quantize_latents
        self.stochastic = stochastic
        self.ls = ls # layer scale: integer factor
        self.input_size = input_size # specified by dataset

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers, sigma=2/L)
        if self.stochastic:
            # Make alpha be less than half the quantization interval
            # Else if not quantizing, make alpha default value of 0.25
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)
            else:
                raise ValueError('Quant. disabled')

        ilw = int(self.ls*32) # initial layer width
        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(ilw, 2*ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

        )
        self.conv_flat_dim = ilw*2*7*7
        self.final = nn.Sequential(
            nn.Linear(self.conv_flat_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.latent_dim),
            nn.Tanh(),
        )

    def forward(self, x, u):
        x = self.main(x)
        x = x.view(-1, self.conv_flat_dim)
        x = self.final(x)

        # in universal quantization, add noise then quantize
        if self.stochastic:
            x = x + u

        if self.quantize_latents:
            x = self.q(x)

        return x
    
class Decoder_mnist(nn.Module):
    def __init__(self, latent_dim, output_size, stochastic):
        super(Decoder_mnist, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.stochastic = stochastic

        self.expand = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        if self.output_size == 784:
            self.l1 = nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0)
            self.l2 = nn.BatchNorm2d(64)
            self.l3 = nn.LeakyReLU()
            self.l4 = nn.ConvTranspose2d(64, 128, kernel_size=5, stride=2, padding=0)
            self.l5 = nn.BatchNorm2d(128)
            self.l6 = nn.LeakyReLU()
            self.l7 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=1, padding=0)
            self.l8 = nn.Sigmoid()
            # )
        else:
            raise ValueError(f'No deconvolution defined for output size of {self.output_size}.')

    def forward(self, x, u):
        if self.stochastic:
            x = x-u
        x = self.expand(x)
        x = x.view(-1, 32, 4, 4)
        x = self.l3(self.l2(self.l1(x, output_size = (11, 11))))
        x = self.l6(self.l5(self.l4(x, output_size = (25, 25))))
        x = self.l8(self.l7(x, output_size=(28, 28)))

        return x