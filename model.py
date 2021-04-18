import copy
import math
import numpy as np
import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, feature_size, num_clusters=256, output_dim=1024, normalize_input=True, alpha=1.0, drop_rate=0.5, gating_reduction=8):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.alpha = alpha

        self.bn1 = nn.BatchNorm1d(feature_size)
        self.conv = nn.Conv1d(feature_size, num_clusters,
                              kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, feature_size))

        self.bn2 = nn.BatchNorm1d(feature_size * num_clusters)
        self.drop = nn.Dropout(drop_rate)

        self.fc1 = nn.Linear(feature_size * num_clusters, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim // gating_reduction)
        self.bn4 = nn.BatchNorm1d(output_dim // gating_reduction)
        self.fc3 = nn.Linear(output_dim // gating_reduction, output_dim)
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x, num_frames):
        N, C, T = x.shape[:3]  # (N, C, T)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        x = self.bn1(x)
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x)  # (N, num_clusters, T)
        soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, T)
        soft_assign = soft_assign * frame_mask.unsqueeze(1)

        soft_assign_sum = torch.sum(
            soft_assign, dim=-1, keepdim=True)  # (N, num_clusters, 1)
        # (N, num_clusters, feature_size)
        centervlad = self.centroids * soft_assign_sum

        x_flatten = x.view(N, C, -1)  # (N, feature_size, T)
        # (N, num_clusters, feature_size)
        vlad = torch.bmm(soft_assign, x_flatten.transpose(1, 2))
        vlad -= centervlad  # (N, num_clusters, feature_size)

        # intra-normalization (N, num_clusters, feature_size)
        vlad = F.normalize(vlad, p=2, dim=2)
        # flatten (N, num_clusters * feature_size)
        vlad = vlad.view(x.size(0), -1)
        vlad = self.bn2(vlad)

        vlad = self.drop(vlad)

        activation = self.bn3(self.fc1(vlad))  # (N, output_dim)

        # (N, output_dim // gating_reduction)
        gates = F.relu(self.bn4(self.fc2(activation)))
        gates = torch.sigmoid(self.fc3(gates))  # (N, output_dim)

        activation = activation * gates  # (N, output_dim)
        # L2 normalize (N, num_clusters * feature_size)
        vlad = F.normalize(activation, p=2, dim=1)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(vlad, p=2, dim=1)

        return embedding  # (N, output_dim)

    def encode(self, x, num_frames):
        N, C, T = x.shape[:3]  # (N, C, T)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        x = self.bn1(x)
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x)  # (N, num_clusters, T)
        soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, T)
        soft_assign = soft_assign * frame_mask.unsqueeze(1)

        soft_assign_sum = torch.sum(
            soft_assign, dim=-1, keepdim=True)  # (N, num_clusters, 1)
        # (N, num_clusters, feature_size)
        centervlad = self.centroids * soft_assign_sum

        x_flatten = x.view(N, C, -1)  # (N, feature_size, T)
        # (N, num_clusters, feature_size)
        vlad = torch.bmm(soft_assign, x_flatten.transpose(1, 2))
        vlad -= centervlad  # (N, num_clusters, feature_size)

        # intra-normalization (N, num_clusters, feature_size)
        vlad = F.normalize(vlad, p=2, dim=2)
        return vlad


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, feature_size, num_clusters=64, output_dim=1024, normalize_input=True, expansion=2, groups=8, drop_rate=0.5, gating_reduction=8):
        super(NeXtVLAD, self).__init__()
        self.feature_size = feature_size
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.expansion = expansion
        self.groups = groups

        self.conv1 = nn.Conv1d(
            feature_size, feature_size * expansion, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(feature_size * expansion,
                               groups, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(feature_size * expansion,
                               num_clusters * groups, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_clusters * groups)
        self.centroids = nn.Parameter(torch.rand(
            num_clusters, feature_size * expansion // groups))

        self.bn2 = nn.BatchNorm1d(
            feature_size * expansion // groups * num_clusters)
        self.drop = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(feature_size * expansion //
                             groups * num_clusters, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim // gating_reduction)
        self.bn4 = nn.BatchNorm1d(output_dim // gating_reduction)
        self.fc3 = nn.Linear(output_dim // gating_reduction, output_dim)

    def forward(self, x, num_frames):
        N, C, T = x.shape[:3]  # (N, C, T)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x = self.conv1(x)  # (N, feature_size * expansion, T)
        # attention factor of per group
        attention = torch.sigmoid(self.conv2(x))  # (N, groups, T)
        attention = attention * frame_mask.unsqueeze(1)
        attention = attention.view(N, 1, -1)  # (N, 1, groups * T)
        # calculate activation factor of per group per cluster
        feature_size = self.feature_size * self.expansion // self.groups

        activation = self.conv3(x)  # (N, num_clusters * groups, T)
        activation = self.bn1(activation)
        # reshape of activation
        # (N, num_clusters, groups * T)
        activation = activation.view(N, self.num_clusters, -1)
        # softmax on per cluster
        # (N, num_clusters, groups * T)
        activation = F.softmax(activation, dim=1)
        activation = activation * attention  # (N, num_clusters, groups * T)
        activation_sum = torch.sum(
            activation, dim=-1, keepdim=True)  # (N, num_clusters, 1)
        # (N, num_clusters, feature_size)
        centervlad = self.centroids * activation_sum

        # (N, feature_size, groups * T)
        x_rehaped = x.view(N, feature_size, -1)
        vlad = torch.bmm(activation, x_rehaped.transpose(1, 2)
                         )  # (N, num_clusters, feature_size)
        vlad -= centervlad  # (N, num_clusters, feature_size)

        # intra-normalization (N, num_clusters, feature_size)
        vlad = F.normalize(vlad, p=2, dim=2)
        # flatten (N, num_clusters * feature_size)
        vlad = vlad.view(N, -1)
        vlad = self.bn2(vlad)

        vlad = self.drop(vlad)

        activation = self.bn3(self.fc1(vlad))  # (N, output_dim)

        # (N, output_dim // gating_reduction)
        gates = F.relu(self.bn4(self.fc2(activation)))
        gates = torch.sigmoid(self.fc3(gates))  # (N, output_dim)

        activation = activation * gates  # (N, output_dim)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(activation, p=2, dim=1)

        return embedding  # (N, output_dim)

    def encode(self, x, num_frames):
        N, C, T = x.shape[:3]  # (N, C, T)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x = self.conv1(x)  # (N, feature_size * expansion, T)
        # attention factor of per group
        attention = torch.sigmoid(self.conv2(x))  # (N, groups, T)
        attention = attention * frame_mask.unsqueeze(1)
        attention = attention.view(N, 1, -1)  # (N, 1, groups * T)
        # calculate activation factor of per group per cluster
        feature_size = self.feature_size * self.expansion // self.groups

        activation = self.conv3(x)  # (N, num_clusters * groups, T)
        activation = self.bn1(activation)
        # reshape of activation
        # (N, num_clusters, groups * T)
        activation = activation.view(N, self.num_clusters, -1)
        # softmax on per cluster
        # (N, num_clusters, groups * T)
        activation = F.softmax(activation, dim=1)
        activation = activation * attention  # (N, num_clusters, groups * T)
        activation_sum = torch.sum(
            activation, dim=-1, keepdim=True)  # (N, num_clusters, 1)
        # (N, num_clusters, feature_size)
        centervlad = self.centroids * activation_sum

        # (N, feature_size, groups * T)
        x_rehaped = x.view(N, feature_size, -1)
        vlad = torch.bmm(activation, x_rehaped.transpose(1, 2)
                         )  # (N, num_clusters, feature_size)
        vlad -= centervlad  # (N, num_clusters, feature_size)

        # intra-normalization (N, num_clusters, feature_size)
        vlad = F.normalize(vlad, p=2, dim=2)
        return vlad


class LSTMModule(nn.Module):
    def __init__(self, feature_size=1024, output_dim=1024, nhid=1024, nlayers=2, dropout=0.2):
        super(LSTMModule, self).__init__()

        self.feature_size = feature_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.dropout = dropout

        self.LSTM = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.bn1 = nn.BatchNorm1d(self.feature_size)

    def forward(self, x, num_frames):
        x = self.bn1(x)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        self.LSTM.flatten_parameters()
        output, (h_n, h_c) = self.LSTM(x, None)

        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (batch, 1)
        output = torch.sum(output, dim=-2) / frame_count

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        return embedding

    def encode(self, x, num_frames):
        x = self.bn1(x)  # (N, C, T)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        self.LSTM.flatten_parameters()
        output, (h_n, h_c) = self.LSTM(x, None)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))
        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        return output


class GRUModule(nn.Module):
    def __init__(self, feature_size=1024, output_dim=1024, nhid=1024, nlayers=2, dropout=0.2):
        super(GRUModule, self).__init__()

        self.feature_size = feature_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.dropout = dropout

        self.GRU = nn.GRU(
            input_size=self.feature_size,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.bn1 = nn.BatchNorm1d(self.feature_size)

    def forward(self, x, num_frames):
        x = self.bn1(x)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        self.GRU.flatten_parameters()
        output, h_n = self.GRU(x, None)

        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (batch, 1)
        output = torch.sum(output, dim=-2) / frame_count

        # L2 normalize (N*num_directions, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        return embedding

    def encode(self, x, num_frames):
        x = self.bn1(x)  # (N, C, T)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        self.GRU.flatten_parameters()
        output, (h_n, h_c) = self.GRU(x, None)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))
        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        return output


class TCA(nn.Module):
    def __init__(self, feature_size=1024, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(TCA, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)

        output = output.permute(1, 0, 2)  # (N, T, C)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        return embedding

    def encode(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output = output.permute(1, 0, 2)  # (N, T, C)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))

        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=1024, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, a, p, n, len_a, len_p, len_n):
        """
        Input:
            a: a batch of anchor logits
            p: a batch of positive logits
            n: a bigger batch of negative logits
        Output:
            logits, targets
        """

        if len(n.size()) > 3:
            n = n.view(-1, n.size()[2], n.size()[3])
            len_n = len_n.view(-1, 1)

        # compute query features
        q = self.encoder_q(a, len_a)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            p = self.encoder_k(p, len_p)  # anchors: NxC
            p = F.normalize(p, dim=1)
            k = self.encoder_k(n, len_n)  # keys: kNxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, p]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return hvd.allgather(tensor.contiguous())


class VideoComparator(nn.Module):
    def __init__(self):
        super(VideoComparator, self).__init__()
        self.pad = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.mpool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.mpool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.fconv = nn.Conv2d(128, 1, (1, 1))

    def forward(self, sim_matrix):
        sim = sim_matrix.reshape(1, 1, sim_matrix.size(
        )[-2], sim_matrix.size()[-1])  # (1, 1, m, n)
        sim = self.pad(sim)
        sim = F.relu(self.conv1(sim))
        sim = self.mpool1(sim)
        sim = self.pad(sim)
        sim = F.relu(self.conv2(sim))
        sim = self.mpool2(sim)
        sim = self.pad(sim)
        sim = F.relu(self.conv3(sim))
        sim = self.fconv(sim)
        sim = torch.clamp(sim, -1.0, 1.0) / 2.0 + 0.5
        sim = sim.reshape(sim.size()[-2], sim.size()[-1])
        return sim
