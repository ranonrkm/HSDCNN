import torch
import torch.nn as nn
import torch.nn.functional as F

from hsd_semantic.tools import decompose_network, TreeNode
from hsd_semantic.config import config

class vgg_hsd(nn.Module):
    def __init__(self, ckpt_path=None):
        super(vgg_hsd, self).__init__()
        convModel, linearModel, cl_arr_ind, subnet_cls, root = decompose_network()
        self.features = convModel
        self.classifier = linearModel
        self.subnet_classes = subnet_cls
        self.subnet_mask = []
        for i in range(len(subnet_cls)):
            self.subnet_mask.append(torch.FloatTensor([1 if j in subnet_cls[i] else 0 for j in range(config.DATASET.NUM_CLASSES)]))
        self.indices = torch.LongTensor(cl_arr_ind)
        self.root = root
 
    def forward(self, x):
        node_que = []
        node_que.append(self.root)
        feat_que = []
        feat_que.append(self.features[self.root.val](x))

        while(node_que):
            p = node_que.pop(0)
            x = feat_que.pop(0)
            if p.left is None and p.right is None:
                feat_que.append(x.view(x.size(0), -1))
                continue
            if p.left is not None:
                node_que.append(p.left)
                feat_que.append(self.features[p.left.val](x))
            if p.right is not None:
                node_que.append(p.right)
                feat_que.append(self.features[p.right.val](x))
        
        out = torch.cat([self.classifier[i](feat_que[i]) for i in range(len(feat_que))], 1)
        out = torch.index_select(out, 1, self.indices.cuda())

        return out
        
