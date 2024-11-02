import torch.nn as nn


class SqrHingeLoss(nn.MultiMarginLoss):
    def __init__(self, *args, **kwargs): 
        super(SqrHingeLoss, self).__init__(p=2, *args, **kwargs)

