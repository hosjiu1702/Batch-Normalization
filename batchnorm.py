 """
 Implement the original Batch Normalization.

 reference: Ioffe et.al 2015 - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            (https://arxiv.org/abs/1502.03167)
 """
 import torch
 import torch.nn as nn


 class BatchNorm(nn.Module):
     
     def __init__(self, gamma, beta, eps, momentum):
         super().__init__()
         self.gamma = gamma
         self.beta = beta
         self.eps = eps
         self.momentum = momentum

         self._running_mean = None
         self._running_var = None

    def forward(self, x):
        if self._running_mean is None:
            self._running_mean = torch.zeros(x.shape, dtype=x.dtype)
        
        if self._running_var is None:
            self._running_var = torch.zeros(x.shape, dtype=x.dtype)

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
