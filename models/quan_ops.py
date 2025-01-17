# Adapted from
# https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py and
# https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/dorefa.py#L25

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import namedtuple
from .lut import lut_actual_15, lut_ideal_15


lut_diff = lut_ideal_15 - lut_actual_15


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


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = 0

    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    return scale, zero_point


def quantize_tensor(x, num_bits=8):
    min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


class FakeQuantOp(torch.autograd.Function):  # equivalent to class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8):
        x = quantize_tensor(x, num_bits=num_bits)
        x_qtensor = x
        x = dequantize_tensor(x)
        return x
        # return x, x_qtensor

    @staticmethod
    def backward(ctx, grad_output):
    # def backward(ctx, grad_output, grad_scale, grad_quantised):

        # straight through estimator
        return grad_output, None, None


def mapMultiplierModel(q_x, q_w):
    q_w_t = torch.t(q_w)  # y = x.wT + b

    res = torch.zeros([q_x.size(0), q_w_t.size(1)])
    for i in range(q_x.size(0)):
        for j in range(q_w_t.size(1)):
            res[i][j] = sum(lut_diff[q_x[i, :], q_w_t[:, j]])

    return res


class fake_quantize_fn(nn.Module):  # equivalent to weight_quantize_fn(nn.Module)
    def __init__(self, bit_list):
        super(fake_quantize_fn, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]

    def forward(self, x):
        x = FakeQuantOp.apply(x, self.wbit)  # x = quantized tensor
        return x

    # def forward(self, x):
    #     x, x_qtensor = FakeQuantOp.apply(x, self.wbit)  # x = quantized tensor
    #     return x, x_qtensor


class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n

        # logging.info('qfn.input*n')
        # logging.info(torch.round(input * n))
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
            # logging.info('weight_quantize_fn.E')
            # logging.info(E)

            weight = torch.tanh(x)
            weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            # logging.info('weight_quantize_fn.weight')
            # logging.info(weight)

            weight_q = 2 * qfn.apply(weight, self.wbit) - 1
            # logging.info('weight_quantize_fn.qfn')
            # logging.info(qfn.apply(weight, self.wbit))

            weight_q = weight_q * E
            # logging.info('weight_quantize_fn.weight_q')
            # logging.info(weight)

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


def myconv2d(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Function to process an input with a standard convolution
    """
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


def myconv2d_lut(input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """
    Function to process an input with a standard convolution
    """
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)

    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()

    # unfold_qtensor = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    # inp_qtensor_unf = unfold_qtensor(input_qtensor)
    # w_qtensor_ = weight_qtensor.view(weight_qtensor.size(0), -1).t()

    # logging.info('input (fake quantized): ')
    # logging.info(input)
    # logging.info('weight (fake quantized): ')
    # logging.info(weight)
    # logging.info('input (fake quantized) unfold: ')
    # logging.info(inp_unf)
    # logging.info('weight (fake quantized) unfold: ')
    # logging.info(w_)
    # logging.info('input (fake quantized) unfold.shape: ')
    # logging.info(inp_unf.shape)
    # logging.info('weight (fake quantized) unfold.shape: ')
    # logging.info(w_.shape)
    # logging.info('input qtensor: ')
    # logging.info(input_qtensor)
    # logging.info('weight qtensor: ')
    # logging.info(weight_qtensor)
    # logging.info('input qtensor.shape: ')
    # logging.info(input_qtensor.shape)
    # logging.info('weight qtensor.shape: ')
    # logging.info(weight_qtensor.shape)
    # logging.info('input qtensor (fake quantized) unfold: ')
    # logging.info(inp_qtensor_unf)
    # logging.info('weight qtensor (fake quantized) unfold: ')
    # logging.info(w_qtensor_)
    # logging.info('input (fake quantized) unfold.shape: ')
    # logging.info(inp_qtensor_unf.shape)
    # logging.info('weight (fake quantized) unfold.shape: ')
    # logging.info(w_qtensor_.shape)

    # loss_c = mapMultiplierModel(inp_qtensor_unf.tensor.transpose(1, 2), w_qtensor_.tensor).transpose(1, 2)
    # compensation = inp_qtensor_unf.tensor * w_qtensor_.tensor * loss_c

    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(w_).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)

    # out_unf = out_unf - compensation

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
            # self.quantize_fn = weight_quantize_fn(self.bit_list)
            self.fake_quantize_fn_weight = fake_quantize_fn(self.bit_list)
            # self.fake_quantize_fn_input  = fake_quantize_fn(self.bit_list)

        def forward(self, input, order=None):
            # weight_q = self.quantize_fn(self.weight)
            weight_q = self.fake_quantize_fn_weight(self.weight)
            return myconv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # def forward(self, input, order=None):
        #     # weight, weight_qtensor = self.fake_quantize_fn_weight(self.weight)
        #     # input,  input_qtensor  = self.fake_quantize_fn_input(input)
        #     weight = self.fake_quantize_fn_weight(self.weight)
        #     # input  = self.fake_quantize_fn_input(input)
        #     # conv_res = myconv2d_lut(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups, input_qtensor, weight_qtensor)
        #     return myconv2d_lut(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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
