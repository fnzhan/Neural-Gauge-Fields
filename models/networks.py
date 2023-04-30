import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np


class rgb_decoder(torch.nn.Module):
    def __init__(self, feat_dim, view_pe=6, middle_dim=256):
        super(rgb_decoder, self).__init__()
        self.input_dim = feat_dim + 3 + 2 * view_pe * 3
        self.view_pe = view_pe
        layer1 = torch.nn.Linear(self.input_dim, middle_dim)
        layer2 = torch.nn.Linear(middle_dim, middle_dim)
        layer3 = torch.nn.Linear(middle_dim, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, features, view_dirs):
        indata = [features, view_dirs]
        indata += [positional_encoding(view_dirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)
        return rgb



class Indexto3D(nn.Module):
    def __init__(self, code_size=256, input_dim=256, output_dim=3, hidden_size=256, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        self.code_size = code_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_dim, self.code_size)
        self.linear2 = nn.Linear(self.code_size, self.hidden_neurons)
        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList([nn.Linear(self.hidden_neurons, self.hidden_neurons)
                                          for i in range(self.num_layers)])
        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(self.hidden_neurons, self.output_dim)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        x = self.last_linear(x)
        x = torch.tanh(x)
        return x


class discrete_transform(nn.Module):
    def __init__(self, input_dim=3, output_dim=256, hidden_size=128, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        # self.code_size = code_size
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.hidden_neurons = hidden_size
        self.num_layers = num_layers

        self.linear1 = nn.Linear(input_dim + 2 * input_dim * 10, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)

        init_weights(self.linear1)
        init_weights(self.linear2)

        # self.linear_list = nn.ModuleList(
        #     [nn.Linear(hidden_size * 2, hidden_size * 2) for i in range(self.num_layers)])

        # for l in self.linear_list:
        #     init_weights(l)

        self.last_linear = nn.Linear(hidden_size * 2, output_dim)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x):

        x = self.linear1(torch.cat([x, positional_encoding(x, 10)], dim=-1))
        x = self.activation(x)
        x = self.activation(self.linear2(x))
        # for i in range(self.num_layers):
        #     x = self.activation(self.linear_list[i](x))
        x = self.last_linear(x)
        x = F.softmax(x, dim=-1)
        return x


class continuous_transform(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, hidden_size=256, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_dim + 2 * input_dim * 10, hidden_size)
        # self.linear1 = nn.Linear(self.input_dim, hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)

        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(self.num_layers)])
        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(hidden_size, output_dim)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x):
        # print('*****', x.shape)

        x = self.linear1(torch.cat([x, positional_encoding(x, 10)], dim=-1))
        # x = self.linear1(x)
        x = self.activation(x)
        x_feat = self.linear2(x)
        x = self.activation(x_feat)

        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        x = self.last_linear(x)
        x = torch.tanh(x)
        return x, x_feat


class continuous_reg(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_size=256, num_layers=2):
        """
        template_size: input size
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.linear1 = nn.Linear(self.input_dim, 128)
        self.linear2 = nn.Linear(128, hidden_size)
        init_weights(self.linear1)
        init_weights(self.linear2)

        self.linear_list = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(self.num_layers)])
        for l in self.linear_list:
            init_weights(l)

        self.last_linear = nn.Linear(hidden_size, output_dim)
        init_weights(self.last_linear)

        self.activation = F.relu

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.activation(self.linear2(x))

        for i in range(self.num_layers):
            x = self.activation(self.linear_list[i](x))
        x = self.last_linear(x)
        return x


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [{}] is not found'.format(activation_type))
    return nonlinearity_layer


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        #  norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, num_groups=16, affine=True)
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(it):
            lr_l = 1.0 - max(0, it - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_xavier_multiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[
            1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = get_xavier_multiplier(m, gain)
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))

def init_seq(s, init_type='xavier_uniform'):
    '''initialize sequential model'''
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            init_weights(a, init_type, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            init_weights(a, init_type, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        else:
            init_weights(a, init_type)
    init_weights(s[-1])

def init_weights(net, init_type='xavier_uniform', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier_uniform':
                xavier_uniform_(m, gain)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_seq(s, init_type='xavier_uniform'):
    '''initialize sequential model'''
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            init_weights(a, init_type, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            init_weights(a, init_type, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        else:
            init_weights(a, init_type)
    init_weights(s[-1])


def positional_encoding(positions, freqs):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(...,2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts
