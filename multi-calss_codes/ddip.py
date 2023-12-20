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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
    parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--p_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
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
        self.ac = nn.Softmax(dim=-1)
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
        x = self.ac(x)
        return x


def train(predictor, x_train, optimizer, fea, batch_size, device):
    predictor.train()

    total_loss = 0
    total_examples = 0
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
        f = torch.concat((fea[edge[0]], fea[edge[2]]), -1)
        # f = torch.concat((h[edge[0]], h[edge[1]]), -1)
        out = predictor(f)
        # h2 = bert(cosima[edge[0]], cosima[edge[1]]).to(test.device)
        # out = h2
        # out = predictor(h[edge[0]], h[edge[1]])
        pred = torch.max(out, 1).indices
        preds = torch.concat([preds, pred], dim=0)
        scores = np.append(scores, out.cpu().numpy())

    pisl = np.array(preds.cpu().numpy() == label.cpu().numpy(), dtype=int)
    # print(pisl)
    # np.save(r'data/pisl.npy', pisl)
    accuracy = accuracy_score(label.cpu().numpy(), preds.cpu().numpy())
    precision = precision_score(label.cpu().numpy(), preds.cpu().numpy(), average='macro')
    recall = recall_score(label.cpu().numpy(), preds.cpu().numpy(), average='macro')
    f1 = f1_score(label.cpu().numpy(), preds.cpu().numpy(), average='macro')
    # auc = roc_auc_score(label.cpu().numpy(), scores, multi_class='ovo')
    print('macro-precision:', precision)
    print('macro-recall:', recall)
    print('macro-f1:', f1)
    print("acc:", accuracy)
    # print("auc:", auc)

    return accuracy, precision, recall, f1


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
    with open(r'../data/entities.csv') as fins:
        for fin in fins:
            index, entity = fin.strip().split(',')
            drug[entity] = index
    with open(r'../data/relations.csv') as fins:
        for fin in fins:
            index2, relation = fin.strip().split(',')
            rel[relation] = index2
    with open(r'../data/train0.csv') as fins:
        for line in fins:
            ent1, re, ent2 = line.strip().split(',')
            trainl.append([int(drug[ent1]), int(rel[re]), int(drug[ent2])])
    with open(r'../data/valid0.csv') as fins:
        for line in fins:
            ent1, re, ent2 = line.strip().split(',')
            test.append([int(drug[ent1]), int(rel[re]), int(drug[ent2])])

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    x_train = torch.tensor(trainl).to(device)
    x_test = torch.tensor(test).to(device)
    print(x_train)

    fea = np.load(r'../data/entity_embedding_1000.npy')
    feature = torch.tensor(fea).to(device)

    runs = args.runs
    in_channel = feature.size(1)*2

    predictor = LinkPredictor(in_channel, args.hidden_channels, 86,
                              args.p_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(
        list(predictor.parameters()),
        lr=args.lr)

    for i in range(runs):
        loss = train(predictor, x_train, optimizer, feature, args.batch_size, device)
        # if i % args.eval_steps == 0:
        accuracy, precision, recall, f1 = file_test(predictor, x_test, optimizer, feature, args.batch_size, device)

    Acc, precision, recall, f1 = file_test(predictor, x_test, optimizer, feature, args.batch_size, device)
    with open("result.txt", 'a')as f:
        f.write(f"{Acc}, {precision}, {recall}, {f1}")
        f.write("\n")
    # torch.save(predictor.state_dict(), "data/ddip.pth")
    print(Acc)

if __name__ == '__main__':
    main()

