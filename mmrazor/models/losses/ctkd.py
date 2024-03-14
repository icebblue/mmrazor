import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.logging import MMLogger

class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        
        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)


from torch.autograd import Function
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value


class LinearDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops - 1

        value = (self._max_value - self._min_value) / self._num_loops
        value = i * (-value)

        return value
    

@MODELS.register_module()
class CTKD(nn.Module):
    def __init__(self,
                 t_start: int = 1,
                 t_end: int = 20,
                 cosine_decay: bool = True,
                 decay_max: float = 0.0,
                 decay_min: float = 0.0,
                 decay_loops: int = 0,
                 max_iter: int = 0,
                 reduction: str = 'batchmean',
                 loss_weight: float = 1.0,
                 teacher_detach: bool = True,):
        super(CTKD, self).__init__()

        self.t_start = t_start
        self.t_end = t_end
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach
        self.epoch = 1
        self.iter = 1
        self.max_iter = max_iter

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

        if cosine_decay:
            self.gradient_decay = CosineDecay(max_value=decay_max, min_value=decay_min, num_loops=decay_loops)
        else:
            self.gradient_decay = LinearDecay(max_value=decay_max, min_value=decay_min, num_loops=decay_loops)

        self.mlp = Global_T()

    def forward(self, preds_S, preds_T):

        cos_value = self.gradient_decay.get_value(self.epoch)
        temp = self.mlp(preds_T, preds_S, cos_value)  # (teacher_output, student_output)
        temp = self.t_start + self.t_end * torch.sigmoid(temp)

        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / temp, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / temp, dim=1)
        loss = (temp**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction) 
        
        if self.iter == self.max_iter:
            self.epoch = self.epoch + 1
            self.iter = 1
        else:
            self.iter = self.iter + 1

        return self.loss_weight * loss
        

