from __future__ import division, print_function, unicode_literals
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import json
import random
import time
from torch.autograd import Variable
from torch.optim import Adam
from network import CNN_KIM
from w2v import load_word2vec
import data_helpers


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='API_classify_data_60_t_percent.p',
                    help='Options: eurlex_raw_text.p, API_classify_data(Programweb).p ')
parser.add_argument('--vocab_size', type=int, default=30001, help='vocabulary size')
parser.add_argument('--vec_size', type=int, default=300, help='embedding size')
parser.add_argument('--sequence_length', type=int, default=300, help='the length of documents')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--tr_batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
parser.add_argument('--start_from', type=str, default='', help='')

parser.add_argument('--learning_rate_decay_start', type=int, default=0,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--learning_rate_decay_every', type=int, default=20,
                    help='how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.95,
                    help='how many iterations thereafter to drop LR?(in epoch)')



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

CNN_net = CNN_KIM( args , embedding_weights )
CNN_net = nn.DataParallel(CNN_net).cuda()

current_lr = args.learning_rate
optimizer = Adam(CNN_net.parameters(), lr=current_lr)

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

for epoch in range(args.num_epochs):
    torch.cuda.empty_cache()

    nr_trn_num = X_trn.shape[0]
    nr_batches = int(np.ceil(nr_trn_num / float(args.tr_batch_size)))

    if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
        frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
        decay_factor = args.learning_rate_decay_rate  ** frac
        current_lr = current_lr * decay_factor
    print('\nLearnRate:',current_lr)
    set_lr(optimizer, current_lr)

    CNN_net.train()

    for iteration, batch_idx in enumerate(np.random.permutation(range(nr_batches))):
        start = time.time()
        start_idx = batch_idx * args.tr_batch_size
        end_idx = min((batch_idx + 1) * args.tr_batch_size, nr_trn_num)

        X = X_trn[start_idx:end_idx]
        Y = Y_trn[start_idx:end_idx]

        data = Variable(torch.from_numpy(X).long()).cuda()
        Y    = Variable(torch.from_numpy(Y).float()).cuda()

        optimizer.zero_grad()

        presict_loss      = 0
        # 性能不够，暂时不计算
        # predict_test_Y  = CNN_net(Variable(torch.from_numpy(X_tst).long()).cuda())
        # predict_loss    = BCE_loss(predict_test_Y,Y_tst)


        activations = CNN_net(data)
        loss = nn.BCELoss()(activations, Y)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        done = time.time()
        elapsed = done - start


        print("\rIteration: {}/{} ({:.1f}%)  Loss:{:.5f}  TestLoss:{:.5f}  Time:{:.5f} ".format(
            iteration, nr_batches,
            iteration * 100 / nr_batches,
            loss.item(), presict_loss, elapsed),
            end="")

    torch.cuda.empty_cache()

    if (epoch + 1) > 0:
        checkpoint_path = os.path.join('save', 'model-api-cnn-' + str(epoch + 1) + '.pth')
        torch.save(CNN_net.state_dict(), checkpoint_path)
        print(" model saved to {}".format(checkpoint_path))
