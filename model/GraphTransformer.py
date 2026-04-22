import os
import sys

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import torch
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, ECConv, global_mean_pool, GINConv
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, dropout):
        super(GraphTransformerEncode, self).__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim)
        )

        self.multiHeadAttention = MultiheadAttention(dim_model=self.in_dim, num_heads=self.num_heads)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()

        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, attr_feature, stru_feature, sp_edge_index, sp_value, edge_rel, rel_encoder, spatial_encoder):
        x_norm = self.layernorm1(attr_feature)

        attn_output, attn_weight = self.multiHeadAttention(x_norm, stru_feature, sp_edge_index, sp_value, edge_rel,
                                                           rel_encoder, spatial_encoder)
        attn_output = self.dropout1(attn_output)

        out1 = attn_output + attr_feature

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight


class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super(SpatialEncoding, self).__init__()

        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU()
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap_ = torch.unsqueeze(lap, dim=-1)  ##[n_edges, 1]
        out = self.fnn(lap_)

        return out


class MultiheadAttention(MessagePassing):

    def __init__(self, dim_model, num_heads, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.d_model = dim_model
        self.num_heads = num_heads

        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)

        self.wq_stru = Linear(dim_model, dim_model)
        self.wk_stru = Linear(dim_model, dim_model)

        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()

        self.wq_stru.reset_parameters()
        self.wk_stru.reset_parameters()
        self.dense.reset_parameters()

    def denominator(self, qs, ks):
        all_ones = torch.ones([ks.shape[0]]).to(qs.device)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)
        return torch.einsum("nhm,hm->nh", qs, ks_sum)


    def forward(self, x_attr, x_stru, sp_edge_index, sp_value, edge_rel, rel_encoder, spatial_encoder):

        q_attr = self.wq(x_attr)
        k_attr = self.wk(x_attr)
        v = self.wv(x_attr).view(x_attr.shape[0], self.num_heads, self.depth)

        q_stru = self.wq_stru(x_stru)
        k_stru = self.wk_stru(x_stru)

        rel_embedding = rel_encoder(edge_rel)

        row, col = sp_edge_index

        query_end_attr = q_attr[col].view(sp_edge_index.shape[1], self.num_heads, self.depth)
        key_start_attr = k_attr[row].view(sp_edge_index.shape[1], self.num_heads, self.depth)

        query_end_stru, key_start_stru = q_stru[col], k_stru[row]

        query_end_stru += rel_embedding
        key_start_stru += rel_embedding

        query_end_stru = query_end_stru.view(sp_edge_index.shape[1], self.num_heads, self.depth)
        key_start_stru = key_start_stru.view(sp_edge_index.shape[1], self.num_heads, self.depth)

        edge_attn_num_attr = torch.einsum("ehd,ehd->eh", query_end_attr, key_start_attr)

        edge_attn_num_stru = torch.einsum("ehd,ehd->eh", query_end_stru, key_start_stru)

        edge_attn_num = edge_attn_num_attr + edge_attn_num_stru

        data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(edge_attn_num.shape[-1], dtype=torch.float32)))
        edge_attn_num *= data_normalizer
        edge_attn_bias = spatial_encoder(sp_value)
        edge_attn_num += edge_attn_bias

        attn_normalizer_attr = self.denominator(q_attr.view(x_attr.shape[0], self.num_heads, self.depth),
                                                k_attr.view(x_attr.shape[0], self.num_heads, self.depth))
        attn_normalizer_stru = self.denominator(q_stru.view(x_stru.shape[0], self.num_heads, self.depth),
                                                k_stru.view(x_stru.shape[0], self.num_heads, self.depth))
        edge_attn_dem = attn_normalizer_attr[col] + attn_normalizer_stru[col]
        attention_weight = edge_attn_num / edge_attn_dem

        outputs = []
        for i in range(self.num_heads):
            output_per_head = self.propagate(edge_index=sp_edge_index, x=v[:, i, :], edge_weight=attention_weight[:, i],
                                             size=None)
            outputs.append(output_per_head)

        out = torch.cat(outputs, dim=-1)

        return self.dense(out), attention_weight


class GraphTransformer(torch.nn.Module):
    def __init__(self, layer_num=3, embedding_dim=64, num_heads=4, num_rel=10, dropout=0.2, type='graph'):
        super(GraphTransformer, self).__init__()

        self.type = type

        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)
        self.spatial_encoder = SpatialEncoding(embedding_dim)

        self.encoder = torch.nn.ModuleList()
        for i in range(layer_num - 1):
            self.encoder.append(
                GraphTransformerEncode(num_heads=num_heads, in_dim=embedding_dim, dim_forward=embedding_dim * 2,
                                       dropout=dropout))

    def reset_parameters(self):
        self.rel_encoder.reset_parameters()
        self.spatial_encoder.reset_parameters()
        for e in self.encoder:
            e.reset_parameters()


    def forward(self, attr_feature, stru_feature, data):
        x_attr = attr_feature
        x_stru = stru_feature

        graph_embedding_layer = []
        attn_layer = []
        for graphEncoder in self.encoder:

            x_attr, attn = graphEncoder(x_attr, x_stru, data.sp_edge_index, data.sp_value, data.sp_edge_rel,
                                        self.rel_encoder, self.spatial_encoder)
            graph_embedding_layer.append(x_attr)
            attn_layer.append(attn)

        final_attr_embedding = graph_embedding_layer[-1]

        if self.type == 'graph':
            sub_representation = []
            for index, drug_mol_graph in enumerate(data.to_data_list()):
                sub_embedding = final_attr_embedding[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(final_attr_embedding, batch=data.batch)
        else:
            sub_representation = []
            for index, drug_subgraph in enumerate(data.to_data_list()):
                sub_embedding = final_attr_embedding[(data.batch == index).nonzero().flatten()]
                sub_representation.append(sub_embedding)
            representation = final_attr_embedding[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer