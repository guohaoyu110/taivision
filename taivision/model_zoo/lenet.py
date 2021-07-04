import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url

__all__ = ['LENET', '_lenet', 'lenet']

class LENET(nn.Module):
    def __init__(self,num_classes=10, input_channels = 1):
        super(LENET, self).__init__()
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5, stride=1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, stride=1),
            nn.MaxPool2d(2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def _lenet(num_classes, **kwargs):
    model = LENET(num_classes = num_classes)
    pretrained = False
    if pretrained:
        raise AssertionError('LENET dont have pretrained model')
    return model

def lenet(num_classes = 10,**kwargs):
    r"""LENET model for MNIST from
    `"Gradient-based learning applied to document recognition" <http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf>`_
    Arg:
        num_class (int), number of class, default 10
        pretrained (bool),If True, raise AssertionError('LENET dont have pretrained model')
    """
    return _lenet(num_classes = num_classes)