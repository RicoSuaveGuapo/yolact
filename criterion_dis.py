import torch
from torch.nn.modules.loss import _Loss

class DiscriminatorLoss_Wgan(_Loss):
    '''
    Wasserstein Distance
    '''
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, input, target):
        # take the minus sign for maximum
        return -(torch.mean(input) - torch.mean(target))

class DiscriminatorLoss_Maskrcnn(_Loss):
    '''
    L1 Distance
    '''
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, input, target):
        # take the minus sign for maximum
        return -torch.mean(torch.abs(input - target))

class GeneratorLoss_Maskrcnn(_Loss):
    '''
    L1 Distance
    '''
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, input, target):
        # take the minus sign for maximum
        return torch.mean(torch.abs(input - target))