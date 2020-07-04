import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

from collections import defaultdict

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
    def forward(self, x1, x2):
        return torch.add(x1, x2)

class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()
    def forward(self, inputs):
        return torch.cat(inputs)

class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()
    def forward(self, x1, x2):
        return torch.mul(x1, x2)

def channel_sizing_last(in_channels, out_channels, i, length):
    if ((i+1) < length):
        out_channels = in_channels
    return in_channels, out_channels

def channel_sizing_linear(in_channels, out_channels, i, length):
    channel_update = (out_channels - in_channels)/length

    out_channels = in_channels + int((i+1)*channel_update)
    in_channels += int(i*channel_update)

    #print(in_channels, out_channels, channel_update)
    if i > 0:
        in_channels -= in_channels % 4
    if i+1 < length:
        out_channels -= out_channels % 4
    return in_channels, out_channels

def block_factory(block, *a, **k):
    def block_instance(*b, **l):
        return block(*b, *a, **l, **k)
    return block_instance

class Stage(nn.Module):
    _skip_connections = defaultdict(None, {
        'add': Add, 
    })

    _channel_sizings = defaultdict(None, {
        'last': channel_sizing_last,
        'linear': channel_sizing_linear,
    })

    def __init__(self,
        block, 
        in_channels, out_channels, kernel_size, 
        pooling_block = None,
        stride = 1, padding = 0,
        length = 1, 
        skip_length = 0,
        skip_connection = 'add',
        channel_sizing = 'linear',
    ):
        """ Builds a stage with length blocks.

        stride: stride of last block
        padding: padding of last block
        length: number of block
        skip_length: number of blocks to skip when using residual connections
            If skip_length < 1, no residual connections are made.
        skip_connection: mode of residuals agregation, in {'add'} or nn.Module
        channel_sizing: how to modify the number of channels during the stage
        """
        super(Stage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.length = length

        if self._channel_sizings[channel_sizing] is None:
            raise ValueError(
                """Channel sizing ('{}') not
                supported.""".format(channel_sizing))
        self.channel_sizing = self._channel_sizings[channel_sizing]

        if skip_length > length-1:
            raise ValueError(
                """Length ({}) of stage not enough to skip by 
                skip_length ({}).""".format(length, skip_length))
        self.skip_length = skip_length

        if self._skip_connections[skip_connection] is None:
            raise ValueError(
                """Skip connection ('{}') not
                supported.""".format(skip_connection))
        if skip_length > 0:
            self.skip_connection = self._skip_connections[skip_connection]()

        inner_kernel_size = kernel_size - ((kernel_size+1)%2)
        inner_padding = inner_kernel_size // 2

        if pooling_block is None:
            pooling_block = block

        self.blocks = nn.ModuleList()
        for i in range(length-1):
            self.blocks.extend([
                block(
                    *self.channel_sizing(
                        in_channels, out_channels, i, length),
                    inner_kernel_size, 
                    padding=inner_padding)
            ])

        self.blocks.extend([
            pooling_block(
                *self.channel_sizing(
                    in_channels, out_channels, length-1, length),
                kernel_size,
                stride=stride, padding=padding)
        ])

    def forward(self, x, *args, **kwargs):
        skip = self.skip_length > 0

        for i, block in enumerate(self.blocks):
            if skip and (i % self.skip_length == 0):
                residuals = x

            x = block(x, *args, **kwargs)

            if (skip and 
                    ((i+1) % self.skip_length == 0) and 
                    i < self.length-1): # don't skip on last (pooling) layer
                x = self.skip_connection(residuals, x)

        return x

class ChannelShuffle(nn.Module):
    """
    Mix the groups channels, as introduced in ShuffleNet.

    With a 2D image batch (batch_size, channels, heigth, width) viewed as
    (batch_size, groups, channel_per_group, height, width), 
    mix the group channels by transposing as 
    (batch_size, channel_per_group, groups, height, width).

    Reference: https://arxiv.org/abs/1707.01083
    """
    def __init__(self, groups, symmetries=1):
        super(ChannelShuffle, self).__init__()

        self.groups = groups
        self.symmetries = symmetries

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups

        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, self.groups * channels_per_group, h, w)

        return x

def accuracy(predictions, targets):
    return (torch.max(predictions, 1)[1] == targets).float().mean()

class VariationalEncoder(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(VariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dimension, out_dimension),
            nn.ReLU(),
            #nn.BatchNorm1d(out_dimension, out_dimension),
            #nn.Sigmoid(),
        )
        self.mu = nn.Linear(out_dimension, out_dimension)
        self.log_var = nn.Linear(out_dimension, out_dimension)
        self.reparameterization = Reparameterization()

    def forward(self, x):
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x)
        x = self.reparameterization(mu, log_var)

        return x, mu, log_var

class Reparameterization(nn.Module):
    def __init__(self):
        super(Reparameterization, self).__init__()

    def forward(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        noise = torch.randn_like(std)

        return mu + noise * std


def VAE_KL_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


if __name__ == '__main__':
    stage = Stage(
        nn.Conv2d, 8, 32, 3,
        stride = 2, padding = 0,
        length = 3,
        skip_length = 0,
        skip_connection = 'add',
        channel_sizing = 'linear',
    )

    print(stage)
    torchsummary.summary(stage, input_size=(8, 32, 32))
