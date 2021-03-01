import numpy as np
import scipy.linalg, scipy.spatial.distance
from collections import OrderedDict
from .class_hierarchy import ClassHierarchy


def get_Softlabels(hierarchy_file, beta=20, class_list=None, is_a=False, str_ids=False):

	id_type = str if str_ids else int
	hierarchy = ClassHierarchy.from_file(hierarchy_file, is_a_relations = is_a, id_type = id_type)

	# Determine target classes
	if class_list is not None:
		with open(class_list) as class_file:
			unique_labels = list(OrderedDict((id_type(l.strip().split()[0]), None) for l in class_file if l.strip() != '').keys())

	else:
		unique_labels = [lbl for lbl in hierarchy.nodes if (lbl not in hierarchy.children) or (len(hierarchy.children[lbl]) == 0)]
		if not str_ids:
			unique_labels.sort()
	linear_labels = { lbl : i for i, lbl in enumerate(unique_labels) }

	sem_class_dist = np.zeros((len(unique_labels), len(unique_labels)))
	for i in range(len(unique_labels)):
		for j in range(i + 1, len(unique_labels)):
			sem_class_dist[i,j] = sem_class_dist[j,i] = hierarchy.lcs_height(unique_labels[i], unique_labels[j])
	soft_labels = np.zeros_like(sem_class_dist)
	for i in range(len(unique_labels)):
		soft_labels[i] = np.exp(-beta*sem_class_dist[i])
	
	return soft_labels


