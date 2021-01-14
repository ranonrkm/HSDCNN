# REF: https://github.com/fiveai/making-better-mistakes/blob/master/better_mistakes/trees.py

import pickle
import lzma
import os
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy


class DistanceDict(dict):
    """
    Small helper class implementing a symmetrical dictionary to hold distance data.
    """
    def __init__(self, distances):
        self.distances = {tuple(sorted(t)): v for t, v in distances.items()}

    def __getitem__(self, i):
        if i[0] == i[1]:
            return 0
        else:
            return self.distances[(i[0], i[1]) if i[0]<i[1] else (i[1], i[0])]

    def __setitem__(self, i):
        raise NotImplementedError()


def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node

def load_hierarchy():
    """
    Load the hierarchy corresponding to a given dataset.

    Returns:
        A nltk tree whose labels correspond to wordnet wnids.
    """
    if config.DATASET.NAME in ["tiered-imagenet-84", "tiered-imagenet-224"]:
        fname = os.path.join(config.HIERARCHYROOT, "tiered_imagenet_tree.pkl")
    elif config.DATASET.NAME in ["ilsvrc12", "imagenet"]:
        fname = os.path.join(config.HIERARCHYROOT, "imagenet_tree.pkl")
    elif config.DATASET.NAME in ["inaturalist19-84", "inaturalist19-224"]:
        fname = os.path.join(config.HIERARCHYROOT, "inaturalist19_tree.pkl")
    elif config.DATASET.NAME.lower() == "cifar100":
        fname = os.path.join(config.HIERARCHYROOT, "cifar100_tree.pkl")
    elif config.DATASET.NAME.lower() == "cifar10":
        fname = os.path.join(config.HIERARCHYROOT, "cifar10_tree.pkl")
    else:
        raise ValueError("Unknown dataset {}".format(config.DATASET.NAME))

    with open(fname, "rb") as f:
        return pickle.load(f)


def load_distances():
    """
    Load the distances corresponding to a given hierarchy.
    Returns:
        A dictionary of the form {(wnid1, wnid2): distance} for all precomputed
        distances for the hierarchy
    """
    
