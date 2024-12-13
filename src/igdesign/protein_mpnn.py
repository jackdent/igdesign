# Adapted from https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py
# ProteinMPNN encoder/Decoder layers according to the official github repo
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torchtyping import TensorType

from igdesign.tokenization import AA_TO_IDX


def exists(x):
    return x is not None


def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(
        -1, -1, -1, edges.size(-1)
    )  # equivalent to just .unsqueeze(-1)
    edge_features = torch.gather(edges, 2, neighbors)  # binary adjacency matrix
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def get_nbr_info(X, mask, top_k=30, eps=1e-6):
    """Pairwise euclidean distances"""
    # Convolutional network on NCHW
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)  # pair mask
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)  # pairwise differences
    D = mask_2D * torch.sqrt(
        torch.sum(dX**2, 3) + eps
    )  # masked pairwise euclidean distances

    # Identify k nearest neighbors (including self)
    D_max, _ = torch.max(
        D, -1, keepdim=True
    )  # maximum distance from a given residue to any other residue
    D_adjust = (
        D + (1.0 - mask_2D) * D_max
    )  # Set all invalid distances to be equal to the maximum distance so they won't be included
    D_neighbors, E_idx = torch.topk(
        D_adjust, np.minimum(top_k, X.shape[1]), dim=-1, largest=False
    )  # get K nearest neighbors for each residue (D_neighbors), and idx of that neighbor (E_idx)
    mask_neighbors = gather_edges(
        mask_2D.unsqueeze(-1), E_idx
    )  # binary adjacency matrix
    return D_neighbors, E_idx, mask_neighbors


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(
        h_nodes, E_idx
    )  # collect all neighbors (batch x seq x K x hidden)
    h_nn = torch.cat(
        [h_neighbors, h_nodes], -1
    )  # concat neighbor nodes with associated edges --> batch x seq x K x 2*hidden
    return h_nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class Residual(pl.LightningModule):
    def __init__(self, use_rezero: bool = False, rezero_init: float = 1e-2):
        super(Residual, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * rezero_init) if use_rezero else 1

    def forward(
        self, f_x: TensorType[..., "hidden"], x: TensorType[..., "hidden"]
    ) -> TensorType[..., "hidden"]:
        return self.alpha * f_x + x


class EncLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_in,
        dropout=0.1,
        num_heads=None,
        scale=30,
        use_rezero: bool = True,
    ):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.node_residual = Residual(use_rezero=use_rezero)
        self.pair_residual = Residual(use_rezero=use_rezero)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(
            h_V, h_E, E_idx
        )  # concatted K nearest nodes and associated self-to-node edges
        h_V_expand = h_V.unsqueeze(-2).expand(
            -1, -1, h_EV.size(-2), -1
        )  # repeat nodes: (batch x seq x K x hidden)
        h_EV = torch.cat(
            [h_V_expand, h_EV], -1
        )  # cat self with neighbor nodes and edges: batch x seq x K x 3*hidden
        h_message = self.W3(
            self.act(self.W2(self.act(self.W1(h_EV))))
        )  # create message. Project from 3*hidden to hidden
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = (
            torch.sum(h_message, -2) / self.scale
        )  # sum together all the messages. Divide by scale to normalize.
        h_V = self.norm1(self.node_residual(x=h_V, f_x=self.dropout1(dh)))

        dh = self.dense(h_V)
        h_V = self.norm2(self.node_residual(x=h_V, f_x=self.dropout2(dh)))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(self.pair_residual(x=h_E, f_x=self.dropout3(h_message)))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(
        self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, use_rezero=True
    ):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        self.node_residual = Residual(use_rezero=use_rezero)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(self.node_residual(x=h_V, f_x=self.dropout1(dh)))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(self.node_residual(x=h_V, f_x=self.dropout2(dh)))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        node_dim_in: int,
        pair_dim_in: int,
        dim_hidden: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout=0.1,
        top_k=64,
        use_rezero: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()
        self.top_k = top_k
        self.dim_hidden = dim_hidden
        self.node_project_in = (
            nn.Sequential(nn.LayerNorm(node_dim_in), nn.Linear(node_dim_in, dim_hidden))
            if node_dim_in > 0
            else nn.Identity()
        )
        self.pair_project_in = nn.Sequential(
            nn.LayerNorm(pair_dim_in), nn.Linear(pair_dim_in, dim_hidden)
        )
        self.seq_embedding = nn.Embedding(len(AA_TO_IDX), dim_hidden)
        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(
                    dim_hidden, dim_hidden * 2, dropout=dropout, use_rezero=use_rezero
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(
                    dim_hidden, dim_hidden * 3, dropout=dropout, use_rezero=use_rezero
                )
                for _ in range(num_decoder_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        node_feats,
        pair_feats,
        res_mask,
        ca_coords,
        chain_mask,
        seq_tokens,
        decode_order=None,
    ) -> Tuple[Tensor, Tensor]:
        res_mask, chain_mask = map(lambda x: x.float(), (res_mask, chain_mask))
        b, n, device = *node_feats.shape[:2], node_feats.device
        node_feats = (
            node_feats
            if exists(node_feats)
            else torch.zeros(b, n, self.dim_hidden, device=device)
        )
        node_feats, pair_feats = self.node_project_in(node_feats), self.pair_project_in(
            pair_feats
        )
        _, E_idx, _ = get_nbr_info(
            ca_coords, res_mask, self.top_k
        )  # E_idx: idx of K nearest neighbors for each residue
        pair_feats = gather_edges(
            pair_feats, E_idx
        )  # pair features of K nearest neighbors for each residue

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(res_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = res_mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            node_feats, pair_feats = layer(
                node_feats, pair_feats, E_idx, res_mask, mask_attend
            )

        encoded_node_feats, encoded_pair_feats = node_feats, pair_feats
        # Concatenate sequence embeddings for autoregressive decoder
        embedded_seq = self.seq_embedding(seq_tokens)  # token embeddings, like from LM
        h_ES = cat_neighbors_nodes(
            embedded_seq, encoded_pair_feats, E_idx
        )  # cat embeddings with edge features

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(embedded_seq), encoded_pair_feats, E_idx
        )  # concat zeros and pair feats
        h_EXV_encoder = cat_neighbors_nodes(
            encoded_node_feats, h_EX_encoder, E_idx
        )  # concat node feats with zeros + pair feats

        chain_mask = (
            chain_mask * res_mask
        )  # update chain mask to include missing regions
        randn = torch.rand_like(chain_mask.float())
        decoding_order = (
            torch.argsort((chain_mask + (torch.abs(randn) * 0.5)))
            if not exists(decode_order)
            else decode_order
        )  # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=n
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(n, n, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = res_mask.view([res_mask.size(0), res_mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        decoded_node_feats = encoded_node_feats
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(decoded_node_feats, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            decoded_node_feats = layer(decoded_node_feats, h_ESV, res_mask)

        return decoded_node_feats, h_ESV, None  # res feats, pair feats, coords

    def get_forward_kwargs_from_batch(self, batch) -> Dict[str, Any]:
        return dict(
            res_mask=batch["masks"]["valid"]["residue"],
            ca_coords=batch["coords"][..., 1, :].detach().clone(),
            chain_mask=batch["masks"]["valid"]["residue"],
            seq_tokens=batch["tokenized_sequences"].clone().long(),
            decode_order=batch["decode_order"] if "decode_order" in batch else None,
        )
