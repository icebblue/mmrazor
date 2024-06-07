#!/usr/bin/env python
"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

"""

import torch
from torch.autograd import Variable
from mmrazor.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class SinkhornLoss(nn.Module):
    """Calculate the two-norm loss between the two features.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def cost_matrix(self, x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        return c
    
    def sinkhorn_loss(self, x, y, epsilon, n, niter):
        """
        Given two emprical measures with n points each with locations x and y
        outputs an approximation of the OT cost with regularization parameter epsilon
        niter is the max. number of steps in sinkhorn loop
        """

        x = x.to(x.device)
        y = y.to(y.device)

        C = self.cost_matrix(x, y).to(x.device)  # Move C to CUDA

        mu = torch.full((n,), 1. / n, device=x.device, requires_grad=False)
        nu = torch.full((n,), 1. / n, device=x.device, requires_grad=False)

        # Parameters of the Sinkhorn algorithm.
        rho = 1  # (.5) **2          # unbalanced transport
        tau = -.8  # nesterov-like acceleration
        lam = rho / (rho + epsilon)  # Update exponent
        thresh = 10**(-1)  # stopping criterion

        # Elementary operations .....................................................................
        def ave(u, u1):
            "Barycenter subroutine, used by kinetic acceleration through extrapolation."
            return tau * u + (1 - tau) * u1

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

        def lse(A):
            "log-sum-exp"
            return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = torch.zeros_like(mu), torch.zeros_like(nu), 0.
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

        for i in range(niter):
            u1 = u  # useful to check the update
            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
            # accelerated unbalanced iterations
            # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
            # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
            err = (u - u1).abs().sum()

            actual_nits += 1
            if err < thresh:
                break
        U, V = u, v
        pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
        cost = torch.sum(pi * C)  # Sinkhorn cost

        return cost

    def sinkhorn_normalized(self, x, y, epsilon, n, niter):
        Wxy = self.sinkhorn_loss(x, y, epsilon, n, niter)
        Wxx = self.sinkhorn_loss(x, y, epsilon, n, niter)
        Wyy = self.sinkhorn_loss(x, y, epsilon, n, niter)
        return 2 * Wxy - Wxx - Wyy
        
    def forward(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
    ) -> torch.Tensor:
        
        # print(preds_S.size(), preds_T.size())
        # print(preds_S)
        # # 获取张量的长度
        # length_S = preds_S.size(0)
        # length_T = preds_T.size(0)

        # # 将索引和值分别存储在两个张量中，并使用 unsqueeze 将其转换为二维张量
        # indices_S = torch.arange(length_S).unsqueeze(1).cuda()  # 生成索引张量并添加一个维度
        # values_S = preds_S.unsqueeze(1).cuda()                 # 将原始张量添加一个维度
        # indices_T = torch.arange(length_T).unsqueeze(1).cuda()  # 生成索引张量并添加一个维度
        # values_T = preds_T.unsqueeze(1).cuda()                  # 将原始张量添加一个维度

        # # 将索引张量和值张量水平拼接在一起形成一个二维张量
        # result_S = torch.cat((indices_S, values_S), dim=1)
        # result_T = torch.cat((indices_T, values_T), dim=1)
        # S = preds_S.transpose(0, 1).contiguous()
        # T = preds_T.transpose(0, 1).contiguous()
        epsilon = 0.01
        niter = 100
        num_classes=16
        
        loss = self.sinkhorn_loss(preds_S, preds_S, epsilon, num_classes, niter)
        return self.loss_weight * loss

