 """
 Implement the original Batch Normalization.

 reference: Ioffe et.al 2015 - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            (https://arxiv.org/abs/1502.03167)
 """
import torch
import torch.nn as nn


class BatchNorm(nn.Module):

    def __init__(
            self,
            num_feats: int,
            eps: float = 1.e-5,
            momentum: float = 0.1
            ):
        super().__init__()
        # BN learnable parameters
        self.gamma = nn.Parameter(torch.zeros(num_feats))
        self.beta = nn.Parameter(torch.zeros(num_feats))

        # BN hyperparams
        self.eps = torch.tensor(num_feats * [eps])
        self.momentum = torch.tensor(num_feat * [momentum])

        self._running_mean = torch.zeros(num_feats)
        self._running_var = torch.zeros(num_feats)

   def forward(self, x):
       if self.training:
           mean = torch.mean(x, 0) # mini-batch mean
           var = torch.var(x, 0, unbiased=False) # mini-batch variance

           """ Use exponential moving average for
           estimating statistics (mean & variance) during training phase
           """
           self._running_mean = (1 - momentum) * self._running_mean + momentum * mean
           self._running_var = (1 - momentum) * self._running_var + momentum * var

           x_hat = (x - mean) / torch.sqrt(var + self.eps) # normalize
       else:
           x_hat = (x - self._running_mean) / torch.sqrt(self._running_var + self.eps)

       y = self.gamma * x_hat + self.beta

       return y

