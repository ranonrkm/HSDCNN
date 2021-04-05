from .resnet import *
from .vgg import *
from hsd_semantic.config import config

def is_resnet(name):
    name = name.lower()
    return name.startswith('resnet')

def is_vgg(name):
    name = name.lower()
    return name.startswith('vgg')

def get_network(in_planes=None):
    
    num_classes = config.DATASET.NUM_CLASSES 
    model = None
    name = config.MODEL.NAME
    if is_resnet(name):
        resnet_size = name[6:]
        resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes) #, in_planes=in_planes)
        model = resnet_model
    elif is_vgg(name):
        vgg_size = name[3:]
        vgg_model = vgg_book.get(vgg_size)(num_classes=num_classes)
        model = vgg_model
    else:
        raise ValueError('Unknown model {}, NOT implemented yet.'.format(name))
    return model


