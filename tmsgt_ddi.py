import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from torch.nn import BCEWithLogitsLoss, Linear
import math
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from .GraphTransformer import GraphTransformer
import os


class NodeFeatures(torch.nn.Module):

    def __init__(self, degree, input_dim, embedding_dim, total_nodes=400000, layer=2, type='graph'):
        super(NodeFeatures, self).__init__()
        self.type = type

        if type == 'graph':
            self.node_encoder = Linear(input_dim, embedding_dim)
        else:

            self.base_embedding = torch.nn.Embedding(total_nodes, embedding_dim)

            self.feature_proj = Linear(input_dim, embedding_dim)

            self.norm = torch.nn.LayerNorm(embedding_dim)

        self.degree_encoder = torch.nn.Embedding(degree, embedding_dim, padding_idx=0)
        self.apply(lambda module: init_params(module, layers=layer))

    def reset_parameters(self):
        self.degree_encoder.reset_parameters()

        if self.type == 'graph':
            self.node_encoder.reset_parameters()
        else:

            self.base_embedding.reset_parameters()
            self.feature_proj.reset_parameters()
            self.norm.reset_parameters()

    def forward(self, data):
        row, col = data.edge_index
        x_degree = degree(col, data.x.size(0), dtype=data.x.dtype)

        if self.type == 'graph':

            attr_feature = self.node_encoder(data.x.float())
        else:

            if hasattr(data, 'node_id'):
                base_emb = self.base_embedding(data.node_id)
            else:

                base_emb = 0

            feat_emb = self.feature_proj(data.x.float())

            attr_feature = base_emb + feat_emb
            attr_feature = self.norm(attr_feature)

        stru_feature = self.degree_encoder(x_degree.long())

        return attr_feature, stru_feature

class TMSGT_DDI(torch.nn.Module):
    def __init__(self, max_layer=6,
                 num_features_drug=67,
                 num_features_bkg=768,
                 num_nodes=391116,
                 num_relations_mol=10,
                 num_relations_graph=10,
                 output_dim=64,
                 max_degree_graph=100,
                 max_degree_node=100,
                 sub_coeff=0.2,
                 mi_coeff=0.5,
                 dropout=0.2,
                 device='cuda'):
        super(TMSGT_DDI, self).__init__()

        print("TMSGT_DDI Loaded")
        self.device = device
        self.mol_coeff = sub_coeff
        self.mi_coeff = mi_coeff

        self.mol_atom_feature = NodeFeatures(degree=max_degree_graph, input_dim=num_features_drug,
                                             embedding_dim=output_dim, type='graph')

        self.drug_node_feature = NodeFeatures(
            degree=max_degree_node,
            input_dim=num_features_bkg,
            embedding_dim=output_dim,
            total_nodes=num_nodes,
            type='node'
        )

        self.mol_representation_learning = GraphTransformer(layer_num=max_layer, embedding_dim=output_dim, num_heads=4,
                                                            num_rel=num_relations_mol, dropout=dropout, type='graph')
        self.node_representation_learning = GraphTransformer(layer_num=max_layer, embedding_dim=output_dim, num_heads=4,
                                                             num_rel=num_relations_graph, dropout=dropout, type='node')

        self.fc1 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(output_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        self.disc = Discriminator(output_dim)
        self.b_xent = BCEWithLogitsLoss()

    def to(self, device):
        self.mol_atom_feature.to(device)
        self.drug_node_feature.to(device)
        self.mol_representation_learning.to(device)
        self.node_representation_learning.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.disc.to(device)
        self.b_xent.to(device)

    def reset_parameters(self):
        self.mol_atom_feature.reset_parameters()
        self.drug_node_feature.reset_parameters()
        self.mol_representation_learning.reset_parameters()
        self.node_representation_learning.reset_parameters()

    def forward(self, drug1_mol, drug1_subgraph, drug2_mol, drug2_subgraph):

        mol1_atom_attr, mol1_atom_stru = self.mol_atom_feature(drug1_mol)
        mol2_atom_attr, mol2_atom_stru = self.mol_atom_feature(drug2_mol)

        drug1_node_attr, drug1_node_stru = self.drug_node_feature(drug1_subgraph)
        drug2_node_attr, drug2_node_stru = self.drug_node_feature(drug2_subgraph)

        mol1_graph_emb, mol1_atom_emb, _ = self.mol_representation_learning(mol1_atom_attr, mol1_atom_stru, drug1_mol)
        mol2_graph_emb, mol2_atom_emb, _ = self.mol_representation_learning(mol2_atom_attr, mol2_atom_stru, drug2_mol)

        drug1_node_emb, drug1_sub_emb, _ = self.node_representation_learning(drug1_node_attr, drug1_node_stru,
                                                                             drug1_subgraph)
        drug2_node_emb, drug2_sub_emb, _ = self.node_representation_learning(drug2_node_attr, drug2_node_stru,
                                                                             drug2_subgraph)

        drug1_embedding = self.fc1(torch.cat([drug1_node_emb, mol1_graph_emb], dim=-1))
        drug2_embedding = self.fc1(torch.cat([drug2_node_emb, mol2_graph_emb], dim=-1))

        score = self.fc2(torch.cat([drug1_embedding, drug2_embedding], dim=-1))

        loss_s_m = self.loss_MI(self.MI(drug1_embedding, mol1_atom_emb)) + self.loss_MI(
            self.MI(drug2_embedding, mol2_atom_emb))
        loss_s_d = self.loss_MI(self.MI(drug1_embedding, drug1_sub_emb)) + self.loss_MI(
            self.MI(drug2_embedding, drug2_sub_emb))

        predicts_drug = F.log_softmax(score, dim=-1)
        loss_label = F.nll_loss(predicts_drug, drug1_mol.y.view(-1))

        loss = loss_label + self.mol_coeff * loss_s_m + self.mi_coeff * loss_s_d

        return torch.exp(predicts_drug)[:, 1], loss

    def MI(self, graph_embeddings, sub_embeddings):
        idx = torch.arange(graph_embeddings.shape[0] - 1, -1, -1)
        idx[len(idx) // 2] = idx[len(idx) // 2 + 1]
        shuffle_embeddings = torch.index_select(graph_embeddings, 0, idx.to(self.device))
        c_0_list, c_1_list = [], []
        for c_0, c_1, sub in zip(graph_embeddings, shuffle_embeddings, sub_embeddings):
            c_0_list.append(c_0.expand_as(sub))
            c_1_list.append(c_1.expand_as(sub))
        c_0, c_1, sub = torch.cat(c_0_list), torch.cat(c_1_list), torch.cat(sub_embeddings)
        return self.disc(sub, c_0, c_1)

    def loss_MI(self, logits):
        num_logits = logits.shape[0] // 2
        temp = torch.rand(num_logits)
        lbl = torch.cat([torch.ones_like(temp), torch.zeros_like(temp)], dim=0).float().to(self.device)
        return self.b_xent(logits.view([1, -1]), lbl.view([1, -1]))

    def save(self, path):
        save_path = os.path.join(path, self.__class__.__name__ + '.pt')
        torch.save(self.state_dict(), save_path)
        return save_path

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None: sc_1 += s_bias1
        if s_bias2 is not None: sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        return logits

def init_params(module, layers=2):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)