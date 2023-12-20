import numpy as np
import pandas as pd
import random
import torch

def get_k_fold_data(k, i, X):
    assert k > 1
    fold_size = X.shape[0] // k
    # print(fold_size)

    X_train = None
    X_valid = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) if j < k-1 else slice(j * fold_size, (j + 1) * fold_size+4)
        if i == 4 and j == 3:
            idx = slice(j * fold_size, (j + 1) * fold_size+4)
        if i == 4 and j == i:
            idx = slice(j * fold_size+4, (j + 1) * fold_size+4)

        X_part = X.iloc[idx, :]
        # print(X_part)
        if j == i:
            X_valid = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = pd.concat([X_train, X_part], ignore_index=True, axis=0)
            # print(X_train.size(),X_valid.size())
    return X_train, X_valid


if __name__ == "__main__":
    # data1 = pd.read_csv(r'drugbank_from_GMPNN.tab', sep="\t")
    data2 = pd.read_table(r'Drugbank_from_sumDDI.txt', sep="\t")
    # data = pd.read_csv(r'../../datasetv3/ChChSe-Decagon_polypharmacy.csv')
    print(data2)
    entities = {}
    rel = {}
    num = 0
    hl = []
    rl = []
    tl = []
    with open(r'Drugbank_from_sumDDI.txt') as fin:
        for line in fin:
            if num > 0:
                # print(line.strip().split('\t'))
                h, t, r = line.strip().split('\t')
                if h not in entities.keys():
                    entities[h] = len(entities)
                if t not in entities.keys():
                    entities[t] = len(entities)
                if r not in rel.keys():
                    rel[r] = len(rel)
                hl.append(entities[h])
                rl.append(rel[r])
                tl.append(entities[t])
            # print(h, t, r)
            num += 1
    # print(entities)
    # print(hl)
    ent = pd.DataFrame({'0': entities.values(), '1': entities.keys()})
    re = pd.DataFrame({'0': rel.values(), '1': rel.keys()})
    # print(ent)
    ent.to_csv(r'entities.csv', index=False, header=False)
    re.to_csv(r'relations.csv', index=False, header=False)
    ddi = pd.DataFrame({0: list(data2['drug1']), 1: list(data2['Label']), 2: list(data2['drug2'])})
    # print(ddi)
    # for i in range(data2.shape[0]):
    ddi = ddi.sample(frac=1).reset_index(drop=True)
    ddi.to_csv(r'ddi.csv', index=False, header=False)
    for i in range(5):
        train, valid = get_k_fold_data(5, i, ddi)
        print(train.shape, valid.shape)
        train.to_csv(f'train{i}.csv', index=False, header=False)
        valid.to_csv(f'valid{i}.csv', index=False, header=False)

