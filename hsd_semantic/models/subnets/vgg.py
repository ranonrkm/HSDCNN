import torch
import torch.nn as nn
import torch.nn.functional as F

from hsd_semantic.tools import decompose_network, TreeNode
from hsd_semantic.config import config

class vgg_hsd(nn.Module):
    def __init__(self, ckpt_path=None):
        super(vgg_hsd, self).__init__()
        convModel, linearModel, cl_arr_ind, subnet_cls, subnets, root = decompose_network()
        self.features = convModel
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = linearModel
        self.subnet_classes = subnet_cls
        self.subnet_mask = []
        for i in range(len(subnet_cls)):
            self.subnet_mask.append(torch.FloatTensor([1 if j in subnet_cls[i] else 0 for j in range(config.DATASET.NUM_CLASSES)]))
        self.subnets = subnets
        self.indices = torch.LongTensor(cl_arr_ind)
        self.root = root
 
    def forward(self, x, subnet_id=None):

        if config.SOLVER.SEPARATE and subnet_id is not None:
            subnet = self.subnets[subnet_id]
            for layer_id in subnet:
                x = self.features[layer_id](x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            out = self.classifier[subnet_id](x)
            return out
            """
            class_indices = torch.tensor(self.subnet_classes[subnet_id])
            mask = torch.eye(config.DATASET.NUM_CLASSES).long()
            mask = torch.index_select(mask, 1, class_indices).cuda()
            return torch.squeeze(mask @ out.unsqueeze(-1), dim=-1)
            """ 
        node_que = []
        node_que.append(self.root)
        feat_que = []
        feat_que.append(self.features[self.root.val](x))

        while(node_que):
            p = node_que.pop(0)
            x = feat_que.pop(0)
            if not p.children:
                feat_que.append(x.view(x.size(0), -1))
                continue
            for child in p.children:
                node_que.append(child)
                feat_que.append(self.features[child.val](x))
            '''
            if p.left is not None:
                node_que.append(p.left)
                feat_que.append(self.features[p.left.val](x))
            if p.right is not None:
                node_que.append(p.right)
                feat_que.append(self.features[p.right.val](x))
            '''
        out = torch.cat([self.classifier[i](feat_que[i]) for i in range(len(feat_que))], 1)
        out = torch.index_select(out, 1, self.indices.cuda())

        return out
        
