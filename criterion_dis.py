import torch
from torch.nn.modules.loss import _Loss

class DiscriminatorLoss(_Loss):
    '''
    Wasserstein Distance
    '''
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, input, target):
        # take the minus sign for maximum
        return -(torch.mean(input) - torch.mean(target))

