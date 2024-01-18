import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util_functions import get_data_split, get_acc, setup_seed, use_cuda, sparse_mx_to_torch_sparse_tensor
from util_functions import load_data_set, symmetric_normalize_adj
from lazy_random_walk_utils import get_lrw_pre_calculated_feature_list
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as sp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = use_cuda()
setup_seed(42)

class DGPN(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(DGPN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h, bias=True)
        self.fc_local_pred_csd = nn.Linear(n_h, 128, bias=True)
        self.fc_final_pred_csd = nn.Linear(n_h, 128, bias=True)
        self.dropout = dropout
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_h)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
    def forward(self, feature_list, csdmatrix):
        temp_embedds = self.fc1(feature_list.to(device))
        temp_embedds = self.act(temp_embedds)
        temp_embedds = F.dropout(temp_embedds, p=self.dropout, training=self.training)
        total_embedds_pred_csd = self.fc_final_pred_csd(temp_embedds)
        total_embedds_pred_csd = F.dropout(total_embedds_pred_csd, p=self.dropout, training=self.training)
        preds = torch.mm(total_embedds_pred_csd, csdmatrix.t())
        return preds, total_embedds_pred_csd

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def feature_propagation(adj, features, k, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    adj = adj.to(device)
    features_prop = features.clone()
    for i in range(1, k + 1):
        features_prop = torch.sparse.mm(adj, features_prop)
        features_prop = (1 - alpha) * features_prop + alpha * features
    return features_prop.cpu()

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def process_adj(adj):
    adj.setdiag(1)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def train(args):
    [c_train, c_val] = args.train_val_class
    idx, labellist, G, features, csd_matrix = load_data_set(args.dataset)
    G = symmetric_normalize_adj(G)
    adj = sparse_mx_to_torch_sparse_tensor(G)
    x = features
    features = feature_propagation(adj, x, args.k, args.alpha)
    idx_train, idx_test, idx_val = get_data_split(c_train=c_train, c_val=c_val, idx=idx, labellist=labellist)
    y_true = np.array([int(temp[0]) for temp in labellist]) #[n, 1]
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)
    model = DGPN(n_in=features.shape[1], n_h=args.n_hidden, dropout=args.dropout).to(device)
    csd_matrix = csd_matrix.to(device)
    domain_classifier = DomainClassifier(128, 128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(list(model.parameters()) + list(domain_classifier.parameters()), lr=args.lr, weight_decay=args.wd)
    result_dir = 'result/'
    result_file = open(file=result_dir + args.dataset + '.txt', mode='a')
    print(args, file=result_file)
    result_file.flush()
    for epoch in range(args.n_epochs+1):
        model.train()
        optimiser.zero_grad()
        preds, embeddings = model(features, csd_matrix)
        loss_global = criterion(preds[idx_train], y_true[idx_train])
        p = [torch.zeros_like(embeddings[0]) for _ in range(c_train)]
        class_counts = [0] * c_train
        for i in idx_train:
            label = y_true[i]
            p[label] += embeddings[i].to(device)
            class_counts[label] += 1
        for i in range(c_train):
            if class_counts[i] > 0:
                p[i] /= class_counts[i]
        metric_loss = 0
        for i in range(c_train):
            for j in range(i + 1, c_train):
                dist = -torch.norm(p[i] - p[j])
                metric_loss += dist
        adv_loss = -(
                torch.log(domain_classifier(embeddings[idx_train])).sum() / len(idx_train)
                + torch.log(1 - domain_classifier(embeddings[idx_test])).sum() / len(idx_test)
        )
        loss = loss_global + adv_loss * 0.001 + metric_loss * 0.00001
        if epoch % 1 == 0:
            train_acc = get_acc(preds[idx_train], y_true[idx_train], c_train=c_train, c_val=c_val, model='train')
            test_acc = get_acc(preds[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test')
            if len(idx_val) != 0:
                val_acc = get_acc(preds[idx_val], y_true[idx_val], c_train=c_train, c_val=c_val, model='val')
                print(epoch, 'Loss:', loss.item(), 'Train_acc:', train_acc, 'Val_acc:', val_acc, 'Test_acc:', test_acc)
                print(epoch, 'Loss:', loss.item(), 'Train_acc:', train_acc, 'Val_acc:', val_acc, 'Test_acc:', test_acc, file=result_file)
                result_file.flush()
            else:
                print(epoch, 'Loss:', loss.item(), 'Train_acc:', train_acc, 'Test_acc:', test_acc)
                print(epoch, 'Loss:', loss.item(), 'Train_acc:', train_acc, 'Test_acc:', test_acc, file=result_file)
                result_file.flush()
            model.eval()
            preds, _ = model(features, csd_matrix)
            test_acc = get_acc(preds[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test') 
            print('Evaluation!', 'Test_acc:', test_acc, "+++")
            print('Evaluation!', 'Test_acc:', test_acc, "+++", file=result_file)
            result_file.flush()
        loss.backward()
        optimiser.step()
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='citeseer', choices=['cora', 'citeseer', 'C-M10-M', 'dblp'], help="dataset")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--train-val-class", type=int, nargs='*', default=[2, 0], help="the first #train_class and #validation classes")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5000, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64, help="number of hidden layers")
    parser.add_argument("--wd", type=float, default=0, help="Weight for L2 loss")
    parser.add_argument("--k", type=int, default=10, help="k-hop neighbors for feature_propagation")
    parser.add_argument("--alpha", type=float, default=0.01, help="hyper-parameter for feature_propagation")
    parser.add_argument("--knn", type=float, default=15, help="hyper-parameter for adj_knn")
    args = parser.parse_args()
    print(args)
    import time
    t = time.time()
    train(args)
    print(time.time() - t)
    print("total time...")
