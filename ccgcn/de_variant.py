import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_
from compgcn_conv import CompGCNConv
from utils import *


class CCGCN(torch.nn.Module):
    def __init__(self, dataset, params):

        super(CCGCN, self).__init__()
        self.dataset = dataset
        self.params = params
        self.dim_t = self.params.t_emb_dim
        self.is_gcn = params.is_gcn
        self.device = params.device

        # number of entities and relations
        self.num_rels = self.dataset.numRel()
        self.num_ents = self.dataset.numEnt()

        # dropout rate for layers
        self.inp_drop = self.params.inp_drop
        self.fin_drop = self.params.fin_drop
        self.hid_drop = self.params.hid_drop
        self.gcn_drop = self.params.gcn_drop

        # size of the layers
        self.gcn_inp_size = params.gcn_inp_dim
        self.gcn_hid_size = params.gcn_hid_dim
        self.gcn_fin_size = params.s_emb_dim

        # create the entity and relation embeddings for the real and imaginary parts
        self.init_rel_r = get_param(
            (self.num_rels * 2, self.gcn_inp_size), device=self.device
        )
        self.init_rel_i = get_param(
            (self.num_rels * 2, self.gcn_inp_size), device=self.device
        )
        self.init_ent_r = get_param(
            (self.num_ents, self.gcn_inp_size), device=self.device
        )
        self.init_ent_i = get_param(
            (self.num_ents, self.gcn_inp_size), device=self.device
        )

        # create the gcn layers
        if self.is_gcn:
            self.conv1 = CompGCNConv(
                self.gcn_inp_size,
                self.gcn_hid_size,
                self.num_rels,
                opn=params.opn,
                act=torch.tanh,
                hid_drop=self.hid_drop,
            )

            self.conv2 = CompGCNConv(
                self.gcn_hid_size,
                self.gcn_fin_size,
                self.num_rels,
                opn=params.opn,
                act=torch.tanh,
                hid_drop=self.hid_drop,
            )

        # create the time embeddings
        self.create_time_embedding()

        # input dropout layer & batch norm layer
        self.input_drop_layer = torch.nn.Dropout(self.inp_drop)
        self.gcn_drop_layer = torch.nn.Dropout(self.gcn_drop)
        self.fin_bn = nn.BatchNorm1d(self.gcn_fin_size + self.dim_t)

    def create_time_embedding(self):
        n = self.num_ents + 2 * self.num_rels
        # frequency embeddings for the entities and relations
        self.d_freq = get_embeding(n, self.dim_t)
        self.m_freq = get_embeding(n, self.dim_t)
        self.y_freq = get_embeding(n, self.dim_t)
        self.i_freq = get_embeding(n, self.dim_t)

        # phase embeddings for the entities and relations
        self.d_phi = get_embeding(n, self.dim_t)
        self.m_phi = get_embeding(n, self.dim_t)
        self.y_phi = get_embeding(n, self.dim_t)
        self.i_phi = get_embeding(n, self.dim_t)

        # amplitude embeddings for the entities and relations
        self.d_amp = get_embeding(n, self.dim_t)
        self.m_amp = get_embeding(n, self.dim_t)
        self.y_amp = get_embeding(n, self.dim_t)
        self.i_amp = get_embeding(n, self.dim_t)

    def get_time_embedding(
        self, obj, years, months, days, intervals, obj_type="ent", comp="r"
    ):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)
        # intervals = intervals.view(-1, 1)

        obj = obj if obj_type == "ent" else obj + self.num_ents

        y_amp, m_amp, d_amp = self.y_amp(obj), self.m_amp(obj), self.d_amp(obj)
        y_freq, m_freq, d_freq = self.y_freq(obj), self.m_freq(obj), self.d_freq(obj)
        y_phi, m_phi, d_phi = self.y_phi(obj), self.m_phi(obj), self.d_phi(obj)
        dynamic_func = lambda freq, comp, phi: freq * comp + phi
        y_dynamics = dynamic_func(y_freq, years, y_phi)
        m_dynamics = dynamic_func(m_freq, months, m_phi)
        d_dynamics = dynamic_func(d_freq, days, d_phi)

        comp_time_func = lambda x: torch.cos(x) if comp == "r" else torch.sin(x)

        feat_func = lambda amp, dynamics: amp * comp_time_func(dynamics)

        emb_temporal = (
            feat_func(y_amp, y_dynamics)
            + feat_func(m_amp, m_dynamics)
            + feat_func(d_amp, d_dynamics)
        )

        return emb_temporal

    def get_diachronic_embedding(self, obj, y, m, d, i, embeddings, obj_type, comp):
        target_emb = torch.index_select(embeddings, 0, obj).to(self.device)
        time_embedding = self.get_time_embedding(obj, y, m, d, i, obj_type, comp)
        time_entity_embedding = torch.cat((target_emb, time_embedding), dim=1)

        return time_entity_embedding

    def complex_embedding(self, obj, y, m, d, i, embeddings, obj_type="ent"):
        emb_r, emb_i = embeddings
        ent_r = self.get_diachronic_embedding(
            obj, y, m, d, i, emb_r, obj_type, comp="r"
        )
        ent_i = self.get_diachronic_embedding(
            obj, y, m, d, i, emb_i, obj_type, comp="i"
        )

        ent_r = self.input_drop_layer(ent_r)
        ent_i = self.input_drop_layer(ent_i)

        return (ent_r, ent_i)

    def pipe_gcn_layers(self, edge_index, edge_type):
        complex_drop = lambda x: (self.gcn_drop_layer(x[0]), self.gcn_drop_layer(x[1]))

        r_init = (self.init_rel_r, self.init_rel_i)
        x_init = (self.init_ent_r, self.init_ent_i)

        x, r = self.conv1(x_init, edge_index, edge_type, rel_embed=r_init)
        x = complex_drop(x)

        x, r = self.conv2(x, edge_index, edge_type, rel_embed=r)
        x = complex_drop(x)

        # scut residual connection
        # x = (x[0] + x_init[0], x[1] + x_init[1])
        # r = (r[0] + r_init[0], r[1] + r_init[1])

        return x, r

    def forward(
        self,
        heads,
        relations,
        tails,
        years,
        months,
        days,
        intervals,
        edge_index,
        edge_type,
    ):
        ent_emb, rel_emb = self.pipe_gcn_layers(edge_index, edge_type)

        # get embeddings for entities and relations
        h = self.complex_embedding(
            heads, years, months, days, intervals, ent_emb, obj_type="ent"
        )
        t = self.complex_embedding(
            tails, years, months, days, intervals, ent_emb, obj_type="ent"
        )
        r = self.complex_embedding(
            relations, years, months, days, intervals, rel_emb, obj_type="rel"
        )

        # get components
        hr, hi = h
        tr, ti = t
        rr, ri = r

        rrr = rr * hr * tr
        rii = rr * hi * ti
        iri = ri * hr * ti
        iir = ri * hi * tr

        scores = rrr + rii + iri - iir
        scores = F.dropout(scores, p=self.fin_drop, training=self.training)
        scores = torch.sum(scores, dim=1)

        return scores
