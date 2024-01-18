import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_functions import dot_sim, use_cuda

device = use_cuda()

from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GATmodel(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''

    def __init__(self, in_feature, hidden_feature):
        super(GATmodel, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.build()
        print("GATConv init.")

    def build(self):
        self.gat_layer_1 = GATConv(in_channels=self.in_feature, out_channels=self.hidden_feature, heads=3, concat=False)
        self.gat_layer_2 = GATConv(in_channels=self.hidden_feature, out_channels=self.hidden_feature, heads=3,
                                   concat=False)

    def forward(self, x, edge_index):
        gout_1 = self.gat_layer_1(x, edge_index)
        gout_2 = F.dropout(F.relu(gout_1), p=0.2, training=self.training)
        gout_2 = self.gat_layer_2(gout_2, edge_index)
        return gout_2


class GCNmodel(torch.nn.Module):
    def __init__(self, in_feature, hidden_feature):
        super(GCNmodel, self).__init__()
        self.conv1 = GCNConv(in_feature, 512)
        self.conv2 = GCNConv(512, hidden_feature)
        self.conv3 = nn.Linear(hidden_feature, hidden_feature)
        self.drop = nn.Dropout(0.2)
        self.act = nn.PReLU()

    def forward(self, x, edge_index):
        gout_1 = self.conv1(x, edge_index)
        gout_2 = F.dropout(self.act(gout_1), p=0.2, training=self.training)
        gout = self.conv2(gout_2, edge_index)
        return gout


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes, bias)

    def forward(self, x):
        return self.linear(x)


class DGPN(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(DGPN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h, bias=True)
        self.fc_local_pred_csd = nn.Linear(n_h, 128, bias=True)
        self.fc_final_pred_csd = nn.Linear(n_h, 128, bias=True) 

        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, feature_list, csdmatrix):
        templist = []
        local_pred_result_list = []
        for features in feature_list:
            temp_embedds = self.fc1(features.to(device))
            temp_embedds = self.act(temp_embedds)
            temp_embedds = F.dropout(temp_embedds, p=self.dropout, training=self.training)
            templist.append(temp_embedds)
            local_pred_csd = self.fc_local_pred_csd(temp_embedds)
            local_pred_result_list.append(dot_sim(local_pred_csd, csdmatrix.t()))

        total_embedds = torch.sum(torch.stack(templist), dim=0)
        total_embedds_pred_csd = self.fc_final_pred_csd(total_embedds)
        total_embedds_pred_csd = F.dropout(total_embedds_pred_csd, p=self.dropout, training=self.training)
        preds = dot_sim(total_embedds_pred_csd, csdmatrix.t())

        return preds, local_pred_result_list
