from __future__ import division, print_function, unicode_literals
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from network import CNN_KIM,XML_CNN
import random
import time
from utils import evaluate,evaluate_xin
import data_helpers
import scipy.sparse as sp
from w2v import load_word2vec
import os
import json

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
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--ts_batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--start_from', type=str, default='save', help='')
parser.add_argument('--re_ranking', type=int, default=80, help='The number of re-ranking size')

parser.add_argument('--kim', type=int, default=1, help='whether evaluate CNN_KIM')
parser.add_argument('--kim_name_root', type=str, default='model-api-cnn-',help='full name is (root+id+".pth") ')
parser.add_argument('--kim_id_begin', type=int, default=1)
parser.add_argument('--kim_id_end', type=int, default=30)

parser.add_argument('--xml', type=int, default=1, help='whether evaluate XML_CNN')
parser.add_argument('--xml_name_root', type=str, default='model-api-xml-cnn-',help='full name is (root+id+".pth") ')
parser.add_argument('--xml_id_begin', type=int, default=1)
parser.add_argument('--xml_id_end', type=int, default=30)


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

nr_tst_num = X_tst.shape[0]
nr_batches = int(np.ceil(nr_tst_num / float(args.ts_batch_size)))
n, k_trn = Y_trn.shape
m, k_tst = Y_tst.shape
print ('k_trn:', k_trn)
print ('k_tst:', k_tst)

Y_tst_pred = np.zeros(Y_tst.shape)

if args.kim !=0:
    for kim_cnn_id in range(args.kim_id_begin, args.kim_id_end+1):
        model_name = "{}{}.pth".format(args.kim_name_root,kim_cnn_id)
        baseline = CNN_KIM(args, embedding_weights)
        baseline = nn.DataParallel(baseline).cuda()
        baseline.load_state_dict(torch.load(os.path.join(args.start_from, model_name)))
        print(model_name + ' loaded')

        for batch_idx in range(nr_batches):
            start = time.time()
            start_idx = batch_idx * args.ts_batch_size
            end_idx = min((batch_idx + 1) * args.ts_batch_size, nr_tst_num)
            X = X_tst[start_idx:end_idx]
            Y = Y_tst_o[start_idx:end_idx]
            data = Variable(torch.from_numpy(X).long()).cuda()

            candidates = baseline(data)
            candidates = candidates.data.cpu().numpy()
            Y_tst_pred[start_idx:end_idx] = candidates

            done = time.time()
            elapsed = done - start

            print("\r Reranking: {} Iteration: {}/{} ({:.1f}%)  Elapsed:{:.5f}".format(
                args.re_ranking, batch_idx, nr_batches,
                batch_idx * 100 / nr_batches,
                elapsed),
                end="")

        print(elapsed)

        evaled = evaluate_xin(Y_tst_pred, Y_tst)
        with open(os.path.join('analyse','cnn_kim.txt'),'a',encoding='utf-8') as f:
            print(model_name,*evaled,sep='\t',file=f)
        del(baseline)

if args.xml !=0:
    for xml_cnn_id in range(args.xml_id_begin, args.xml_id_end+1):
        model_name = "{}{}.pth".format(args.xml_name_root,xml_cnn_id)
        baseline = XML_CNN(args, embedding_weights)
        baseline = nn.DataParallel(baseline).cuda()
        baseline.load_state_dict(torch.load(os.path.join(args.start_from, model_name)))
        print(model_name + ' loaded')

        for batch_idx in range(nr_batches):
            start = time.time()
            start_idx = batch_idx * args.ts_batch_size
            end_idx = min((batch_idx + 1) * args.ts_batch_size, nr_tst_num)
            X = X_tst[start_idx:end_idx]
            Y = Y_tst_o[start_idx:end_idx]
            data = Variable(torch.from_numpy(X).long()).cuda()

            candidates = baseline(data)
            candidates = candidates.data.cpu().numpy()
            Y_tst_pred[start_idx:end_idx] = candidates

            done = time.time()
            elapsed = done - start

            print("\r Reranking: {} Iteration: {}/{} ({:.1f}%)  Elapsed:{:.5f}".format(
                args.re_ranking, batch_idx, nr_batches,
                batch_idx * 100 / nr_batches,
                elapsed),
                end="")

        print(elapsed)

        evaled = evaluate_xin(Y_tst_pred, Y_tst)
        with open(
                os.path.join('analyse','xml_cnn.txt'),
                'a',encoding='utf-8') \
                as f:
            print(model_name,*evaled,sep='\t',file=f)
        del(baseline)

