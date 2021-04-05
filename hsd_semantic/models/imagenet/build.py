import torch.nn as nn
from torchvision import models
from hsd_semantic.config import config
from types import MethodType

def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        feat = self.classifier[-1][:-2](self.classifier[:-1](x))
        out = self.classifier[-1][-2:](feat)
        return out

def get_network(args):
    n_classes = config.DATASET.NUM_CLASSES
    model = models.__dict__[config.MODEL.NAME](pretrained=args.pretrained)
    if config.MODEL.NAME.lower().startswith('vgg'):
        feature_dim = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
                nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, n_classes))

        model.forward = MethodType(forward, model)

    elif  config.MODEL.NAME.lower().startswith('resnet'):
        feature_dim = model.fc.in_features
        model.fc = nn.Sequential(torch.nn.Dropout(0.2), 
                    torch.nn.Linear(in_features=feature_dim, 
                                    out_features=n_classes, 
                                    bias=True))
    return model
