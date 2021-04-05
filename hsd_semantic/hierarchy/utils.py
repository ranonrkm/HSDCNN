import os
from hsd_semantic.config import config

def decode(filename):
    class_dict = {}
    file1 = open(filename, 'r')
    lines = file1.readlines() 
    for line in lines:
        Id, class_name = line.split()
        class_dict[Id] = class_name
    file1.close()
    return class_dict

def get_clusters(clusters_encoded):
    filename = os.path.join(config.HIERARCHY.ROOT, 
                            config.DATASET.NAME + "-Hierarchy", 
                            "class_names.txt")
    class_dict = decode(filename)
    class_clusters = {}
    for i, cluster in enumerate(clusters_encoded):
        class_clusters[i] = [class_dict[str(Id)] for Id in cluster]

    return class_clusters
     
