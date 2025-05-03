from torch.optim.optimizer import Optimizer
import torch
from . import StiefelParameter, SPDParameter
       
def orthogonal_projection(A, B):
    out = A - B @ A.transpose(-2, -1) @ B
    return out


def retraction(A, ref=None):
    if ref is None:
        data = A
    else:
        data = A + ref
    Q, R = data.qr()
    sign = (R.diagonal(dim1=-2, dim2=-1).sign() + 0.5).sign().diag_embed()
    out = Q @ sign
    return out

def exp(X):
    S, U = torch.linalg.eigh(X)
    S = S.exp().diag_embed()
    return U @ S @ U.transpose(-2, -1)

def expm(x, y):
    c = torch.linalg.cholesky(x)
    c_inv = c.inverse()
    return c @ exp(c_inv @ y @ c_inv.transpose(-2, -1)) @ c.transpose(-2, -1)

class StiefelMetaOptimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0
                if isinstance(p, StiefelParameter):
                    trans = orthogonal_projection(p.grad, p)
                    p.grad.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    riem = p @ ((p.grad + p.grad.transpose(-2, -1)) / 2) @ p
                    self.state[p] = p.clone()
                    p.fill_(0)
                    p.grad.fill_(0).add_(riem)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p)
                    p.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    trans = expm(self.state[p], p)
                    p.fill_(0).add_(trans)

        return loss


class MyOptimizer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0

        loss = self.optimizer.step(closure)
        return loss
