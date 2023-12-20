import argparse
import random
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve
# from gin import GIN
# from pretrain_smiles_embedding import get_str_feature, graph_construction_and_featurization

EMB_INIT_EPS = 2.0
gamma = 12.0


def parse_GATv2_DDI_args():
    parser = argparse.ArgumentParser(description="GATv2_DDI")
    parser.add_argument('--seed', type=int, default=2023, help='Random seed.')
    parser.add_argument('--entity_dim', type=int, default=400, help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=4, help='Relation Embedding size.')
    parser.add_argument('--aggregation_type', nargs='?', default='sum')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]')
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5, help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--DDI_l2loss_lambda', type=float, default=1e-5, help='Lambda when calculating DDI l2 loss.')
    # parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    # parser.add_argument('--n_epoch', type=int, default=200, help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10, help='Number of epoch for early stopping')
    parser.add_argument('--multi_type', nargs='?', default='False', help='whether task is multi-class')
    parser.add_argument('--n_hidden_1', type=int, default=128, help='FC hidden 1 dim')
    parser.add_argument('--n_hidden_2', type=int, default=128, help='FC hidden 2 dim')
    parser.add_argument('--out_dim', type=int, default=1, help='FC output dim: 81 or 1')
    parser.add_argument('--structure_dim', type=int, default=300, help='structure_dim')
    parser.add_argument('--pre_entity_dim', type=int, default=400, help='pre_entity_dim')
    parser.add_argument('--device', type=int, default=3)
    # parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--p_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=100)
    # parser.add_argument("--type", dest="type", default=GAT_TYPE.GAT2, type=GAT_TYPE.from_string, choices=list(GAT_TYPE))
    parser.add_argument('--use_layer_norm', action='store_true', default=True)
    parser.add_argument('--use_residual', action='store_true', default=True)
    parser.add_argument('--use_resdiual_linear', action='store_true', default=False)
    parser.add_argument('--num_heads', type=int, default=8)

    args = parser.parse_args()

    return args


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        # self.lins = torch.nn.ModuleList()
        # self.concat = nn.Sequential(nn.Linear(in_channels_2, in_channels), nn.BatchNorm1d(in_channels),
        #                             nn.ReLU(True))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        # self.lins.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            # self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.BatchNorm1d(hidden_channels))
            hidden2_channels = hidden_channels
            hidden_channels = hidden_channels // 2
            self.lins.append(torch.nn.Linear(hidden2_channels, hidden_channels))
        self.batchnorm = nn.BatchNorm1d(hidden_channels)
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        # x_i = self.concat(x_i)
        # x = x_i + x_jz
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        # return torch.sigmoid(x)
        x = self.softmax(x)
        return x



def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train = None
    X_valid = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = X[idx, :]
        if j == i:
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            # print(X_train.size(),X_valid.size())
    return X_train, X_valid


def train(predictor, x_train, optimizer, fea, batch_size, device):
    predictor.train()

    total_loss = total_examples = 0
    idx = TensorDataset(torch.LongTensor(range(x_train.size(0))))
    for perm in DataLoader(idx, batch_size,
                           shuffle=False):
        optimizer.zero_grad()
        edge = x_train[perm].t()
        x = fea
        f = torch.concat((x[edge[0]], x[edge[2]]), -1)
        out = predictor(f)

        label_l = torch.tensor(range(len(edge[1])))
        loss = -torch.log(out[label_l, edge[1]] + 1e-15).mean()
        # print("batch_loss:", loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        # torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def file_test(predictor, test, optimizer, fea, batch_size, device):
    predictor.eval()

    label = test.t()[1]
    preds = torch.tensor([], dtype=torch.int64).to(test.device)
    idx = TensorDataset(torch.LongTensor(range(test.size(0))))
    scores = np.array([])
    for perm in DataLoader(idx, batch_size,
                           shuffle=False):
        edge = test[perm].t()
        labe = torch.LongTensor(range(edge[1].size(0)))
        f = torch.concat((fea[edge[0]], fea[edge[2]]), -1)
        # f = torch.concat((h[edge[0]], h[edge[1]]), -1)
        out = predictor(f)
        # h2 = bert(cosima[edge[0]], cosima[edge[1]]).to(test.device)
        # out = h2
        # out = predictor(h[edge[0]], h[edge[1]])
        pred = torch.max(out, 1).indices
        preds = torch.concat([preds, pred], dim=0)
        score = out[:, 1]
        # score = out
        # print(score)
        scores = np.append(scores, score.cpu().numpy())
    # print(label.cpu().numpy())

    print(preds)
    accuracy = accuracy_score(label.cpu().numpy(), preds.cpu().numpy())
    precision = precision_score(label.cpu().numpy(), preds.cpu().numpy())
    recall = recall_score(label.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy())
    # aucc = calAUC(scores, label.cpu().numpy())
    # aucc = roc_auc_score(preds.cpu().numpy(), scores ,multi_class='ovo')
    p, r, t = precision_recall_curve(y_true=label.cpu().numpy(), probas_pred=scores)
    aupr = auc(r, p)
    fpr, tpr, thresholds = roc_curve(label.cpu().numpy(), scores, pos_label=1)
    print(thresholds)
    aucc = auc(fpr, tpr)
    print('precision:', precision)
    print('auc:', aucc)
    print('aupr:', aupr)
    print('f1:', f1)
    print("acc:", accuracy)

    return accuracy

def calAUC(prob,labels):
  f = list(zip(prob,labels))
  rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
  rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
#   print(rankList)
  posNum = 0
  negNum = 0
  for i in range(len(labels)):
    if(labels[i]==1):
      posNum+=1
    else:
      negNum+=1
  auc = 0
  auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
#   print(auc)
  return auc

def get_matrix(x, y):
    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    mtr_l = x * y.permute((0, 2, 1))
    # print(mtr_l.size())
    mtr_l = mtr_l.view(mtr_l.size(0), -1)
    # mtr_r = y * x.permute((0, 2, 1))
    # print(mtr_l.size())
    return mtr_l

def main():
    print("hello")
    args = parse_GATv2_DDI_args()
    SEED = 2023
    random.seed(SEED)
    print(args)
    # kg = np.load(r'./dataset/use_data_2/id_transE.npz')
    # smi = np.load(r'./dataset/use_data_2/use_smiles.npy')
    # sms = np.load(r'dataset/use_data_2/sms.npy')
    # sms = torch.tensor(sms, dtype=torch.float32)
    # entity_pre_embed = torch.tensor(kg['transE'])
    # structure_pre_embed = torch.tensor(smi)
    # n_entities = len(smi)
    # n_relations = args.relation_dim
    drug = {}
    rel = {}
    trainl = []
    test = []
    with open(r'../entities.csv') as fins:
        for fin in fins:
            entity, index = fin.strip().split(',')
            drug[entity] = len(drug)
    with open(r'../relations.csv') as fins:
        for fin in fins:
            relation, index2 = fin.strip().split(',')
            rel[relation] = len(rel)
    with open(r'../train0.csv') as fins:
        for line in fins:
            ent1, re, ent2 = line.strip().split(',')
            trainl.append([drug[ent1], rel[re], drug[ent2]])
    # print(rel)
    # print(drug)
    with open(r'../valid0.csv') as fins:
        for line in fins:
            ent1, re, ent2 = line.strip().split(',')
            # print(int(drug[ent2]))
            test.append([int(drug[ent1]), int(rel[re]), int(drug[ent2])])
    # print(trainl)
    # trainl = np.array(trainl)
    # test = np.array(test)SS
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device2 = torch.device(device)
    # device = torch.device('cuda:0')
    x_train = torch.tensor(trainl).to(device)
    x_test = torch.tensor(test).to(device)
    print(x_train)
    fea = np.load(r'../../bi_result/entity_embedding0.npy')
    # st = np.load(r'data/use_smiles_concat.npy')
    feature = torch.tensor(fea).to(device)
    # str = torch.tensor(st).to(device)
    # print(dataset)

    runs = args.runs

    in_channel = feature.size(1)*2
    # gin = GIN(num_node_emb_list=[120, 3], num_edge_emb_list=[6, 3], num_layers=5, emb_dim=300,
    #           JK='last', dropout=0.5).to(device)
    predictor = LinkPredictor(in_channel, args.hidden_channels, 2,
                              args.p_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(
        list(predictor.parameters()),
        lr=args.lr)

    for i in range(runs):
        loss = train(predictor, x_train, optimizer, feature, args.batch_size, device)
        if i % args.eval_steps == 0:
            acc = file_test(predictor, x_test, optimizer, feature, args.batch_size, device)

    Acc = file_test(predictor, x_test, optimizer, feature, args.batch_size, device)
    print(Acc)

if __name__ == '__main__':
    main()

