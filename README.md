
# Introduction

This repository contains the implementaion for **Taxonomy Driven Learning Of Semantic Hierarchy Of Classes** published in IEEE International Conference on Image Processing (ICIP), 2022. The link to the paper: https://ieeexplore.ieee.org/abstract/document/9898007

## Setup

```
pip install -e .
export DATA_PATH= <your_data_path_root>
export IMPACT_PATH= <root_for_impact_scores>
```

## Create Impact Scores

* TODO

## Checking Generated Clusters

```
python check_clusters.py
```

## Train

* Train pretrained network
```
    python main.py
```
* Train hsd-cnn
```
    python main.py -H
```

