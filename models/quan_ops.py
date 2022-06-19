# Adapted from
# https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py and
# https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/dorefa.py#L25

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, bit_list):
        super(SwitchBatchNorm2d, self).__init__()
        self.bit_list = bit_list
        self.bn_dict = nn.ModuleDict()
        for i in self.bit_list:
            self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)

        self.abit = self.bit_list[-1]
        self.wbit = self.bit_list[-1]
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x


def batchnorm2d_fn(bit_list):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, bit_list=bit_list):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, bit_list=bit_list)

    return SwitchBatchNorm2d_


class SwitchBatchNorm1d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, bit_list):
        super(SwitchBatchNorm1d, self).__init__()
        self.bit_list = bit_list
        self.bn_dict = nn.ModuleDict()
        for i in self.bit_list:
            self.bn_dict[str(i)] = nn.BatchNorm1d(num_features)

        self.abit = self.bit_list[-1]
        self.wbit = self.bit_list[-1]
        if self.abit != self.wbit:
            raise ValueError('Currenty only support same activation and weight bit width!')

    def forward(self, x):
        x = self.bn_dict[str(self.abit)](x)
        return x


def batchnorm1d_fn(bit_list):
    class SwitchBatchNorm1d_(SwitchBatchNorm1d):
        def __init__(self, num_features, bit_list=bit_list):
            super(SwitchBatchNorm1d_, self).__init__(num_features=num_features, bit_list=bit_list)

    return SwitchBatchNorm1d_


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2**k - 1)
        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class weight_quantize_fn(nn.Module):
    def __init__(self, bit_list):
        super(weight_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        assert self.wbit <= 8 or self.wbit == 32

    def forward(self, x):
        if self.wbit == 32:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q = weight * E
        else:
            E = torch.mean(torch.abs(x)).detach()
            logging.info('weight_quantize_fn.E')
            logging.info(E)

            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            logging.info('weight_quantize_fn.weight')
            logging.info(weight)

            weight_q = 2 * qfn.apply(weight, self.wbit) - 1
            logging.info('weight_quantize_fn.qfn')
            logging.info(qfn.apply(weight, self.wbit))

            weight_q = weight_q * E
            logging.info('weight_quantize_fn.weight_q')
            logging.info(weight)

        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, bit_list):
        super(activation_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.abit = self.bit_list[-1]
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            activation_q = qfn.apply(x, self.abit)
        return activation_q


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)


def myconv2d(input, weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1):
    """
    Function to process an input with a standard convolution
    """
    logging.info('input')
    logging.info(input)
    logging.info('weight')
    logging.info(weight)

    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()
    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(w_).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()


def conv2d_quantize_fn(bit_list):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.bit_list = bit_list
            self.w_bit = self.bit_list[-1]
            self.quantize_fn = weight_quantize_fn(self.bit_list)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return myconv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_Q_


class Linear_Q(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(Linear_Q, self).__init__(*kargs, **kwargs)


def linear_quantize_fn(bit_list):
    class Linear_Q_(Linear_Q):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q_, self).__init__(in_features, out_features, bias)
            self.bit_list = bit_list
            self.w_bit = self.bit_list[-1]
            self.quantize_fn = weight_quantize_fn(self.bit_list)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)

    return Linear_Q_


batchnorm_fn = batchnorm2d_fn
