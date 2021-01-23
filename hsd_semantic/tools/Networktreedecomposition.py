import os
import numpy as np
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import * #fclusterdata,linkage,complete,dendrogram
#from sklearn.cluster import bicluster as Bicluster
from sklearn.cluster import AgglomerativeClustering,Birch,SpectralClustering
from scipy.spatial.distance import pdist
from collections import deque
import matplotlib.pyplot as plt
import _pickle as pickle
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from hsd_semantic.models import get_network
from hsd_semantic.config import config

class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def get_layer_scores(layer):
    for class_label in range(config.DATASET.NUM_CLASSES):
        filename = os.path.join(os.environ['IMPACT_PATH'], 
                                config.DATASET.NAME, 
                                config.MODEL.NAME, 
                                str(config.HIERARCHY.BETA), 
                                'class' + str(class_label) + 'in' + str(layer) + '.pkl')
        with open(filename, 'rb') as infile:
            try:
                X = np.asarray(pickle.load(infile))
            except FileNotFoundError:
                print("{} does not exist.".format(filename))
        if class_label == 0:
            orig_imp_score = np.zeros([config.DATASET.NUM_CLASSES, X.shape[0]])
        orig_imp_score[class_label] = np.abs(X).mean(axis=1)
    
    return orig_imp_score

def scores_processed(imp_score):
    imp_normscore = normalize(imp_score, norm='l2', axis=1)
    imp_normscore[imp_normscore == 0] = np.finfo('float').eps
    return imp_normscore

def get_clusters(layer, class_present, flag):

    class_present = np.sort(np.asarray(class_present))
    imp_score = get_layer_scores(layer)
    imp_normscore = scores_processed(imp_score)
    imp_normscore = imp_normscore[class_present, :]

    class_labels = {}
    filter_indices = {}
    if len(class_present) < 2:
        flag = 0
    
    if flag:
        n_clust = 2
        dist_metric = config.CLUSTER.DIST_METRIC
        link_method = config.CLUSTER.LINK_METHOD
        ag = AgglomerativeClustering(n_clusters=n_clust, affinity=dist_metric,
                                    linkage=link_method).fit(imp_normscore).labels_
        for i in set(ag):
            clstr_labels = np.where(ag == i)[0]
            
            if clstr_labels.shape[0] < 2 or clstr_labels.shape[0] > class_present.size - 2:
                flag = 0
                continue
            mask_score = imp_normscore[clstr_labels, :]
            idx = mask_score.argmax(0)
            cols_min = int(np.floor(config.CLUSTER.CPR * idx.shape[0]))
            filters = np.argsort(mask_score[idx,np.arange(idx.shape[0])])[cols_min:]
            clstr_labels = np.sort(class_present[clstr_labels])
            class_labels[i] = list(clstr_labels)
            filter_indices[i] = list(set(filters))

    if flag == 0:
        class_labels[0] = class_present
        mask_score = imp_normscore[:, :]
        idx = mask_score.argmax(0)
        cols_min = int(np.floor(config.CLUSTER.CPR * idx.shape[0]))
        filters = np.argsort(mask_score[idx, np.arange(idx.shape[0])])[cols_min:]
        filter_indices[0] = list(set(filters))

    return class_labels, filter_indices


# function for decomposing network

def decompose_network():
	
    layers = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40] #, 43]
    #decompose_layers = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1] #, 1]  	#flags to decide whether to decompose
    #decompose_layers = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1]
    decompose_layers = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1]
    assert len(layers)==len(decompose_layers)
    N=len(layers)

    layerAtPresent = []
    parent = []
    fmapsPerNode = {}
    classesPerNode = {}	
    imp_score = get_layer_scores(0)
    layerAtPresent.append(0)
    parent.append(-1)
    classesPerNode[0] = list(range(imp_score.shape[0]))
    fmapsPerNode[0] = list(range(imp_score.shape[1]))

    root = TreeNode(0)
    que = deque()
    que.append(root)

    count=0
    flag=0
    while(que):
        current = que.popleft()
        next_layer_id = layerAtPresent[current.val] + 1
        if next_layer_id == N:
            continue
        flag = decompose_layers[next_layer_id]
        next_layer = layers[next_layer_id]
        cls_clstrs, fmap_clstrs = get_clusters(next_layer, classesPerNode[current.val], flag)
        
        for i, clstr in cls_clstrs.items():
            count += 1
            layerAtPresent.append(next_layer_id)
            parent.append(current.val)
            classesPerNode[count] = clstr
            fmapsPerNode[count] = fmap_clstrs[i]
            if i==0:
                current.left = TreeNode(count)
                que.append(current.left)
            else:
                current.right = TreeNode(count)
                que.append(current.right)

    assert len(que)==0
    print(layerAtPresent)
    for key, val in classesPerNode.items():
        print("{} : {}".format(key, val))


    # Creating the dissected layers from the saved checkpoint
    net = get_network(config)
    try:
        net.load_state_dict(torch.load(config.MODEL.CKPT))
    except FileNotFoundError:
        print("{} does not exist.".format(config.MODEL.CKPT))
    except:
        print("{} does not contain checkpoint for the {} model.".format(config.MODEL.CKPT, 
                                                                        config.MODEL.NAME))

    convModel = nn.ModuleList()
    bnModel = nn.ModuleList()
    convModel = nn.ModuleList()
 
    for node_id in range(count+1):
        m = layers[layerAtPresent[node_id]]
        node_name = 'conv2d' + str(m) + ' ' + str(node_id)
        out_channel_ids = fmapsPerNode[node_id]
        if parent[node_id] == -1:
            in_channel_ids = list(range(3))
        else:
            in_channel_ids = fmapsPerNode[parent[node_id]]
        current_layers = []

        l = net._modules['features']._modules.get(str(m))
        if isinstance(l, nn.Conv2d):
            in_channels, out_channels = len(in_channel_ids), len(out_channel_ids)
            bias = True if l.bias is not None else False
    
            copy_layer = nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=l.kernel_size, 
                                    stride=l.stride, 
                                    padding=l.padding,
                                    dilation=l.dilation,
                                    groups=l.groups, 
                                    bias=bias)
            
            copy_layer.weight.data.zero_()
            l_w = l.weight.data[out_channel_ids,:,:,:].clone()
            copy_layer.weight.data = l_w[:,in_channel_ids,:,:]
            if bias:
                copy_layer.bias.data.zero_()
                copy_layer.bias.data = l.bias.data[out_channel_ids].clone()
            
            current_layers.append(copy_layer)
        else:
            raise TypeError("node {} must have a corresponding convolution block in the parent network."
                            .format(node_id))

        l_bn = net._modules['features']._modules.get(str(m+1))
        if isinstance(l_bn, nn.BatchNorm2d):
            num_features = len(out_channel_ids)
            bncopy_layer = nn.BatchNorm2d(num_features, eps=l_bn.eps, 
                                            momentum=l_bn.momentum,
                                            affine=l_bn.affine,
                                            track_running_stats=l_bn.track_running_stats)
            
            bncopy_layer.weight.data = l_bn.weight.data[out_channel_ids].clone()
            bncopy_layer.bias.data = l_bn.bias.data[out_channel_ids].clone()
            if l_bn.running_mean is not None:
                bncopy_layer.running_mean.copy_(l_bn.running_mean[out_channel_ids])
                bncopy_layer.running_var.copy_(l_bn.running_var[out_channel_ids])

            current_layers.append(bncopy_layer)
        else:
            raise TypeError("node {} must have a corresponding batchnorm block in the parent network."
                            .format(node_id))

        current_layers.append(copy.deepcopy(net._modules['features']._modules.get(str(m+2))))

        l_m = net._modules['features']._modules.get(str(m+3))
        if isinstance(l_m, nn.MaxPool2d):
            current_layers.append(copy.deepcopy(l_m))

        convModel.append(nn.Sequential(*current_layers))

    # classifier (FC layers)
    nodesAtLastLayer = list(np.where(np.asarray(layerAtPresent) == len(layers)-1)[0])
    #l_ap = net._modules['features']._modules.get(str(len(layers)))
    #for node in nodesAtLastLayer:
    #    convModel[node].add_module('4',l_ap)

    linearModel = nn.ModuleList()
    classifier = net._modules['classifier']
    for node in nodesAtLastLayer:
        in_channel_ids = torch.LongTensor(fmapsPerNode[node])
        out_channel_ids = torch.LongTensor(classesPerNode[node])

        bias = True if classifier.bias is not None else False
        cp_cl = nn.Linear(len(in_channel_ids), len(out_channel_ids), bias)
        cp_cl.weight.data = (classifier.weight.data[out_channel_ids].clone())[:, in_channel_ids]
        if bias:
            cp_cl.bias.data = classifier.bias.data[out_channel_ids].clone()

        linearModel.append(cp_cl)

    cl_end = []
    subnet_cls = []
    for node in nodesAtLastLayer:
        print("{} : {}".format(node, classesPerNode[node]))
        subnet_cls.append(classesPerNode[node])
        cl_end.extend(list(classesPerNode[node]))

    cl_arr = np.asarray(cl_end).reshape(1, config.DATASET.NUM_CLASSES)[0,:]
    cl_arr_ind = np.argsort(cl_arr)
    print('cl_arr_ind')
    print(cl_arr_ind)

    return convModel, linearModel, cl_arr_ind, subnet_cls, root 
