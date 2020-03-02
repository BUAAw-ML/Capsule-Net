from __future__ import division, print_function, unicode_literals
import torch
from torchvision import models
from torchsummary import summary

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vgg = models.vgg16().to(device)
#
# summary(vgg, (3, 224, 224))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from network import CNN_KIM,CapsNet_Text
import random
import time
from utils import evaluate,evaluate_xin
import data_helpers
import scipy.sparse as sp
from w2v import load_word2vec
import os

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='API_classify_data(Programweb).p',
                    help='Options: eurlex_raw_text.p, API_classify_data(Programweb).p')
parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')
parser.add_argument('--vec_size', type=int, default=300, help='embedding size')
parser.add_argument('--sequence_length', type=int, default=300, help='the length of documents')
parser.add_argument('--is_AKDE', type=bool, default=True, help='if Adaptive KDE routing is enabled')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--ts_batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--start_from', type=str, default='save', help='')

parser.add_argument('--num_compressed_capsule', type=int, default=64, help='The number of compact capsules')
parser.add_argument('--dim_capsule', type=int, default=8, help='The number of dimensions for capsules')

parser.add_argument('--re_ranking', type=int, default=200, help='The number of re-ranking size')

import json
args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))

X_trn, Y_trn, Y_trn_o, X_tst, Y_tst, Y_tst_o, vocabulary, vocabulary_inv = data_helpers.load_data(args.dataset,
                                                                                max_length=args.sequence_length,
                                                                                vocab_size=args.vocab_size)
Y_trn = Y_trn.toarray()
Y_tst = Y_tst.toarray()

X_trn = X_trn.astype(np.int32)
X_tst = X_tst.astype(np.int32)
Y_trn = Y_trn.astype(np.int32)
Y_tst = Y_tst.astype(np.int32)

embedding_weights = load_word2vec('glove', vocabulary_inv, args.vec_size)
args.num_classes = Y_trn.shape[1]

capsule_net = CapsNet_Text(args, embedding_weights)
summary(capsule_net,[300,1])