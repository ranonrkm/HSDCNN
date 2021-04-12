import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# CUDNN
# -----------------------------------------------------------------------------
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'vgg16'
_C.MODEL.CKPT = '/scratch/16ee35016/hsd_semantic/checkpoint/dump/interpretable/cifar10/vgg16/vgg16-1205-10.pth' 
#_C.MODEL.CKPT = '/scratch/16ee35016/HSDCNN/hsd_semantic/checkpoints/cifar100/vgg16/30/model-best-13-03.pth.tar' 
#_C.MODEL.CKPT = '/scratch/16ee35016/HSDCNN/hsd_semantic/checkpoints/cifar10/vgg16/30/model-best-04-03.pth.tar'
_C.MODEL.CKPT_PATH = '/scratch/16ee35016/checkpoints' 
_C.MODEL.HSD_CKPT = ''

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = '/scratch/16ee35016/data' 
_C.DATASET.NAME = 'cifar100'
_C.DATASET.NUM_CLASSES = 100
_C.DATASET.RES = 224
_C.DATASET.AUGMENT = False
# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
_C.CLUSTER = CN()
_C.CLUSTER.DIST_METRIC = 'euclidean' 
_C.CLUSTER.LINK_METHOD = 'ward'
_C.CLUSTER.CPR = 0.5
_C.CLUSTER.NCLUST = 2

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.WORKERS = 4
_C.SOLVER.XE = False
_C.SOLVER.LOSS = 'soft-labels'
_C.SOLVER.NUM_EPOCHS = 200
_C.SOLVER.BASE_LR = 0.1
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.SCHEDULER_FACTOR = 0.1
_C.SOLVER.SCHEDULER_PATIENCE = 8
_C.SOLVER.SEPARATE = False  # for hsd-cnn, consider subnet predictions separately
# -----------------------------------------------------------------------------
# Hierarchy
# -----------------------------------------------------------------------------
_C.HIERARCHY = CN()
_C.HIERARCHY.BETA = 10
_C.HIERARCHY.ROOT = '/scratch/16ee35016/HSDCNN/hsd_semantic/hierarchy'
_C.HIERARCHY.IMPACT_PATH = '/scratch/16ee35016/HSDCNN/hsd_semantic/impact_scores'

def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

"""
if __name__ == "__main__":
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
"""
