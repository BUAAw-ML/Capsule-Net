from __future__ import division, print_function, unicode_literals
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from network import CNN_KIM,XML_CNN
from network import CapsNet_Text
from network import CapsNet_Text_short
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

parser.add_argument('--dataset', type=str, default='API_classify_data_60_t_percent.p',
                    help='Options: eurlex_raw_text.p, API_classify_data(Programweb).p')
parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')
parser.add_argument('--vec_size', type=int, default=300, help='embedding size')
parser.add_argument('--sequence_length', type=int, default=302, help='the length of documents')
parser.add_argument('--is_AKDE', type=bool, default=True, help='if Adaptive KDE routing is enabled')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--ts_batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--start_from', type=str, default='save', help='')

parser.add_argument('--CNN_type', type=int, default=1, help='0:xml_cnn 1:kim_cnn ...')
parser.add_argument('--baseline', type=str, default='model-api-cnn-30.pth', help='use CNN as baseline, default is kim_cnn')
parser.add_argument('--short', type=int, default=1, help='use Cap:0 or Cap_for_short_text:1')
parser.add_argument('--capsule_name_root', type=str, default='model-api-akde-short-',help='full name is (root+id+".pth") ')
parser.add_argument('--capsule_id_begin', type=int, default=21)
parser.add_argument('--capsule_id_end', type=int, default=30)

parser.add_argument('--num_compressed_capsule', type=int, default=64, help='The number of compact capsules')
parser.add_argument('--dim_capsule', type=int, default=8, help='The number of dimensions for capsules')

parser.add_argument('--re_ranking', type=int, default=10, help='The number of re-ranking size')


def transformLabels(labels):
    '''

    :param labels:[ ['1','3'],
                    ['1','3','8']]
    :return:
        ['1', '3', '8']

        [[1. 1. 0.]
         [1. 1. 1.]]
    '''
    label_index = list(set([l for _ in labels for l in _]))
    label_index.sort()

    variable_num_classes = len(label_index)
    target = []
    for _ in labels:
        tmp = np.zeros([variable_num_classes], dtype=np.float32)
        tmp[[label_index.index(l) for l in _]] = 1
        target.append(tmp)
    target = np.array(target)
    return label_index, target


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


nr_tst_num = X_tst.shape[0]
nr_batches = int(np.ceil(nr_tst_num / float(args.ts_batch_size)))
n, k_trn = Y_trn.shape
m, k_tst = Y_tst.shape
print ('k_trn:', k_trn)
print ('k_tst:', k_tst)


model_name = args.baseline
if   args.CNN_type == 0 :
    baseline = XML_CNN(args, embedding_weights)
elif args.CNN_type == 1 :
    baseline = CNN_KIM(args, embedding_weights)
else :
    assert False
baseline = nn.DataParallel(baseline).cuda()
baseline.load_state_dict(torch.load(os.path.join(args.start_from, model_name)))
print(model_name + ' loaded')


if   args.short == 0:
    CapsNet = CapsNet_Text
elif args.short == 1:
    CapsNet = CapsNet_Text_short
else :
    assert False

for id in range(args.capsule_id_begin,args.capsule_id_end):
    capsule_net = CapsNet(args, embedding_weights)
    capsule_net = nn.DataParallel(capsule_net).cuda()
    model_name = "{}{}.pth".fromat(args.capsule_name_root,id)
    capsule_net.load_state_dict(torch.load(os.path.join(args.start_from, model_name)))
    print(model_name + ' loaded')

    capsule_net.eval()
    top_k = 30
    row_idx_list, col_idx_list, val_idx_list = [], [], []
    for batch_idx in range(nr_batches):
        start = time.time()
        start_idx = batch_idx * args.ts_batch_size
        end_idx = min((batch_idx + 1) * args.ts_batch_size, nr_tst_num)
        X = X_tst[start_idx:end_idx]
        Y = Y_tst_o[start_idx:end_idx]
        data = Variable(torch.from_numpy(X).long()).cuda()

        candidates = baseline(data)
        candidates = candidates.data.cpu().numpy()

        Y_pred = np.zeros([candidates.shape[0], args.num_classes])
        for i in range(candidates.shape[0]):
            candidate_labels = candidates[i, :].argsort()[-args.re_ranking:][::-1].tolist()
            _, activations_2nd = capsule_net(data[i, :].unsqueeze(0), candidate_labels)
            Y_pred[i, candidate_labels] = activations_2nd.squeeze(2).data.cpu().numpy()

        for i in range(Y_pred.shape[0]):
            sorted_idx = np.argpartition(-Y_pred[i, :], top_k)[:top_k]
            row_idx_list += [i + start_idx] * top_k
            col_idx_list += (sorted_idx).tolist()
            val_idx_list += Y_pred[i, sorted_idx].tolist()

        done = time.time()
        elapsed = done - start

        print("\r Reranking: {} Iteration: {}/{} ({:.1f}%)  Loss: {:.5f} {:.5f}".format(
            args.re_ranking, batch_idx, nr_batches,
            batch_idx * 100 / nr_batches,
            0, elapsed),
            end="")

    m = max(row_idx_list) + 1
    n = max(k_trn, k_tst)
    print('', elapsed)
    Y_tst_pred = sp.csr_matrix((val_idx_list, (row_idx_list, col_idx_list)), shape=(m, n))

    if k_trn >= k_tst:
        Y_tst_pred = Y_tst_pred[:, :k_tst]

    evaled = evaluate_xin(Y_tst_pred.toarray(), Y_tst)
    with open(os.path.join('analyse', 'capsule.txt'), 'a', encoding='utf-8') as f:
        print(model_name, *evaled, sep='\t', file=f)
    del (capsule_net)
