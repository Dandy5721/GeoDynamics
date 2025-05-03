import torch
from torch import nn
import numpy as np
import os

def log(X):
    S, U = torch.linalg.eigh(X)
    mask = (S <= 0).any(dim=-1)
    if mask.any():
        S_min, _ = S.min(dim=-1)
        S = S + ((1e-5 + abs(S_min)) * mask).unsqueeze(-1)
    S = S.log().diag_embed()
    return U @ S @ U.transpose(-2, -1)


class SPDTangentSpace(nn.Module):
    def __init__(self):
        super(SPDTangentSpace, self).__init__()

    def forward(self, input):
        output = torch.linalg.matrix_log(input)
        # output = log(input)
        return output


class SPDExpMap(nn.Module):
    def __init__(self):
        super(SPDExpMap, self).__init__()

    def forward(self, input):
        s, u = torch.linalg.eigh(input)
        s = s.exp().diag_embed()
        output = u @ s @ u.transpose(-2, -1)

        return output


class SPDConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        epsilon=1e-3,
        padding=None,
        padding_num=1e-4,
    ):
        super(SPDConv, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.v = nn.Parameter(
            torch.randn((out_channels, in_channels) + kernel_size), requires_grad=True
        )
        nn.init.xavier_uniform_(self.v)
        self.padding = padding
        self.padding_num = padding_num
        self.stride = stride

    def forward(self, input):
        if len(input.shape) < 4:
            input = input.unsqueeze(1)

        if self.padding is not None:
            new_shape = list(input.shape)
            new_shape[-2] += 2 * self.padding 
            new_shape[-1] += 2 * self.padding 
            input_new = torch.zeros(new_shape, device=input.device)
            row_idx, col_idx = np.diag_indices(new_shape[-1])
            input_new[..., row_idx, col_idx] = self.padding_num
            input_new[
                ...,
                self.padding : input_new.shape[-2] - self.padding,
                self.padding : input_new.shape[-1] - self.padding,
            ] = input
            input = input_new
        # print("input_new",input.shape)

        weight = self.v.transpose(-2, -1) @ self.v + self.epsilon[0] * torch.eye(
            self.v.shape[-1], device=input.device
        )

        return nn.functional.conv2d(input.float(), weight=weight, stride=self.stride)


class SPDAvgPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SPDAvgPool, self).__init__()
        self.log_map = SPDTangentSpace()
        self.avg_pool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.exp_map = SPDExpMap()

    def forward(self, input):
        input_log = self.log_map(input)
        avg_pool = self.avg_pool(input_log)
        output = self.exp_map(avg_pool)
        return output


class SPDActivate(nn.Module):
    def __init__(self, activate_func='sinh'):
        super(SPDActivate, self).__init__()
        if activate_func == 'sinh':
            self.activate_func = torch.sinh
        elif activate_func == 'cosh':
            self.activate_func = torch.cosh
        else:
            self.activate_func = torch.exp

    def forward(self, input):
        output = self.activate_func(input)
        return output


class SPDDiag(nn.Module):
    def __init__(self):
        super(SPDDiag, self).__init__()

    def forward(self, input):
        output = []
        for block in input:
            output.append(torch.block_diag(*block))
        return torch.stack(output)


class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, input):
        norm = input.norm(self.p, self.dim, keepdim=True)
        output = input / norm
        return output


class SPDAttention(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(SPDAttention, self).__init__()
        self.conv = SPDConv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, input):
        weight = self.conv(input)
        # print(input.shape)
        weight = (weight - weight.max()).exp()
        # print(f"Weight matrix saved to {output_path}")
        output = weight * input

        return output, weight
