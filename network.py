import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import PrimaryCaps, FCCaps, FlattenCaps

def BCE_loss(x, target):
    return nn.BCELoss()(x.squeeze(2), target)

class CapsNet_Text(nn.Module):
    def __init__(self, args, w2v):
        super(CapsNet_Text, self).__init__()
        self.num_classes = args.num_classes
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))

        self.ngram_size = [2,4,8]

        self.convs_doc = nn.ModuleList([nn.Conv1d(args.sequence_length, 32, K, stride=2) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=args.dim_capsule, in_channels=32, out_channels=32, kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()

        self.W_doc = nn.Parameter(torch.FloatTensor(14272, args.num_compressed_capsule))
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(args, output_capsule_num=args.num_classes, input_capsule_num=args.num_compressed_capsule,
                            	  in_channels=args.dim_capsule, out_channels=args.dim_capsule)

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
        # 比如图片img的size比如是（28，28，3）就可以利用img.permute(2, 0, 1)
        # 得到一个size为（3，28，28）的tensor
        # torch.matmul代表两张量的矩阵乘积

        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def forward(self, data, labels):
        data = self.embed(data)
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](data)
            nets_doc_l.append(nets)

        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        # 使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐

        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)

        poses, activations = self.flatten_capsules(poses_doc, activations_doc)

        poses, activations = self.compression(poses, self.W_doc)

        poses, activations = self.fc_capsules_doc_child(poses, activations, labels)

        return poses, activations

class CapsNet_Text_short(nn.Module):
# the most import figure that API data has is:
# short
# so the conv layer should have more out channel
# we can use 128 test it
    def __init__(self, args, w2v):
        super(CapsNet_Text_short, self).__init__()
        self.num_classes = args.num_classes
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))

        self.ngram_size = [3,4,5]

        self.convs_doc = nn.ModuleList([nn.Conv1d(args.sequence_length, 128, K, stride=1) for K in self.ngram_size])
        torch.nn.init.xavier_uniform_(self.convs_doc[0].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[1].weight)
        torch.nn.init.xavier_uniform_(self.convs_doc[2].weight)

        self.primary_capsules_doc = PrimaryCaps(num_capsules=args.dim_capsule, in_channels=128, out_channels=16, kernel_size=1, stride=1)

        self.flatten_capsules = FlattenCaps()

        self.W_doc = nn.Parameter(torch.FloatTensor(7128*2, args.num_compressed_capsule))
        # when pri_cap_doc 's out channel is 8,  W_doc's is 7128
        torch.nn.init.xavier_uniform_(self.W_doc)

        self.fc_capsules_doc_child = FCCaps(args, output_capsule_num=args.num_classes, input_capsule_num=args.num_compressed_capsule,
                            	  in_channels=args.dim_capsule, out_channels=args.dim_capsule)

    def compression(self, poses, W):
        poses = torch.matmul(poses.permute(0,2,1), W).permute(0,2,1)
        # 比如图片img的size比如是（28，28，3）就可以利用img.permute(2, 0, 1)
        # 得到一个size为（3，28，28）的tensor
        # torch.matmul代表两张量的矩阵乘积

        activations = torch.sqrt((poses ** 2).sum(2))
        return poses, activations

    def forward(self, data, labels):
        data = self.embed(data)
        nets_doc_l = []
        for i in range(len(self.ngram_size)):
            nets = self.convs_doc[i](data)
            nets_doc_l.append(nets)

        nets_doc = torch.cat((nets_doc_l[0], nets_doc_l[1], nets_doc_l[2]), 2)
        # 使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐

        poses_doc, activations_doc = self.primary_capsules_doc(nets_doc)

        poses, activations = self.flatten_capsules(poses_doc, activations_doc)

        poses, activations = self.compression(poses, self.W_doc)

        poses, activations = self.fc_capsules_doc_child(poses, activations, labels)

        return poses, activations


class CNN_KIM(nn.Module):

    def __init__(self, args, w2v):
        super(CNN_KIM, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))
        self.conv13 = nn.Conv2d(1, 128, (3, args.vec_size))
        self.conv14 = nn.Conv2d(1, 128, (4, args.vec_size))
        self.conv15 = nn.Conv2d(1, 128, (5, args.vec_size))

        self.fc1 = nn.Linear(3 * 128, args.num_classes)
        self.m = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def loss(self, x, target):
        return nn.BCELoss()(x, target)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x1 = self.conv_and_pool(x,self.conv13)
        x2 = self.conv_and_pool(x,self.conv14)
        x3 = self.conv_and_pool(x,self.conv15)
        x = torch.cat((x1, x2, x3), 1)
        activations = self.fc1(x)
        return self.m(activations)

class XML_CNN(nn.Module):

    def __init__(self, args, w2v):
        super(XML_CNN, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))
        self.conv13 = nn.Conv1d(args.sequence_length, 32, 2, stride=2)
        self.conv14 = nn.Conv1d(args.sequence_length, 32, 4, stride=2)
        self.conv15 = nn.Conv1d(args.sequence_length, 32, 8, stride=2)

        self.fc1 = nn.Linear(14272, 512)
        self.fc2 = nn.Linear(512, args.num_classes)
        self.m = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        return x

    def loss(self, x, target):
        return nn.BCELoss()(x, target)

    def forward(self, x):
        x = self.embed(x)
        batch_size = x.shape[0]

        x1 = self.conv13(x).reshape(batch_size, -1)
        x2 = self.conv14(x).reshape(batch_size, -1)
        x3 = self.conv15(x).reshape(batch_size, -1)
        x = torch.cat((x1, x2, x3), 1)
        hidden = self.fc1(x)
        activations = self.fc2(hidden)
        return self.m(activations)

class XML_CNN_fix(nn.Module):
    # un fixed yet !!!
    def __init__(self, args, w2v):
        super(XML_CNN_fix, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.vec_size)
        self.embed.weight = nn.Parameter(torch.from_numpy(w2v))
        self.conv13 = nn.Conv1d(args.vec_size, 32, 2, stride=2)
        self.conv14 = nn.Conv1d(args.vec_size, 32, 4, stride=2)
        self.conv15 = nn.Conv1d(args.vec_size, 32, 8, stride=2)

        self.fc1 = nn.Linear(14368, 512)
        self.fc2 = nn.Linear(512, args.num_classes)
        self.m = nn.Sigmoid()

    def conv_and_pool(self, x, conv):
        # however there isn't a pool here
        x = F.relu(conv(x))
        return x

    def loss(self, x, target):
        return nn.BCELoss()(x, target)

    def forward(self, x):
        print('input:', x.size())
        x = self.embed(x)
        print('embadded:',x.size())
        x = x.permute(0,2,1)
        print('permuted:',x.size())

        assert False
        batch_size = x.shape[0]

        x1 = self.conv_and_pool(x,self.conv13).reshape(batch_size, -1)
        x2 = self.conv_and_pool(x,self.conv14).reshape(batch_size, -1)
        x3 = self.conv_and_pool(x,self.conv15).reshape(batch_size, -1)
        x = torch.cat((x1, x2, x3), 1)
        hidden = self.fc1(x)
        activations = self.fc2(hidden)
        return self.m(activations)


# what does 14272 mean ?
# when con core step in voc_vectors(vec_size=300)
# output is a matix and size is (((vec_size-core_size) / stride) +1) * output_channels
# so in conv13 conv14 conv15
# all output cat together , size is
# (((300-2)/2+1)+((300-4)/2+1)+((300-8)/2+1))*32 = 14272