import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random

from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

import pickle
import collections

TEMPORAL_VARIATION = "none"
# ['none', 'att', 'pos-add', 'pos-cat', 'tgat-add', 'tgat-cat',
#                              'att-pos-add', 'att-pos-cat', 'att-tgat-add', 'att-tgat-cat']
PROVIDE_EMB = False
EPOCHS = 1
BATCH_SIZE = 8
NEG_SAMPLES = 0  # 5, 25, 50

EVAL_EPOCHS = 1
EVAL_BATCH_SIZE = 1
EVAL_NEG_SAMPLES = 1

LR = 0.03
FANOUTS = [25, ]  # 5, 15, 25
IN_DIM = 100
OUT_DIM = 100  # 100, 200, 400

FEAT_DROP = 0.1  # 0.0, 0.1, 0.2
AGGREGATOR_TYPE = 'mean'  # mean, pool, lstm
NORM = True  # False, True
ACTIVATION = None  # None, F.relu
T_DIM = False

THETA = 1
LOG_DT = False



# device = th.device('cpu')
device = th.device('cuda:0')