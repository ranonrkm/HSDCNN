from .vgg import vgg_hsd
from hsd_semantic.config import config

def is_resnet(name):
    name = name.lower()
    return name.startswith('resnet')

def is_vgg(name):
    name = name.lower()
    return name.startswith('vgg')

def get_hsd():

    model = None
    name = config.MODEL.NAME
    """
    if is_resnet(name):
        resnet_size = name[6:]
        resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes) #, in_planes=in_planes)
        model = resnet_model
    """
    if is_vgg(name):
        model = vgg_hsd()
    else:
        raise ValueError('Unknown model {}, NOT implemented yet.'.format(name))
    return model
