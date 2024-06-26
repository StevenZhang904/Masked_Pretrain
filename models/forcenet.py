"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from math import pi as PI
from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from models.forcenet_util.base import BaseModel
from models.forcenet_util.activations import Act
from models.forcenet_util.basis import Basis, SphericalSmearing
from models.forcenet_util.atomic_radii import ATOMIC_RADII
from models.forcenet_util.continuous_embeddings import CONTINUOUS_EMBEDDINGS


class FNDecoder(nn.Module):
    ### this returns the 3D force of atom.
    def __init__(
        self, decoder_type, decoder_activation_str, output_dim: int
    ) -> None:
        super(FNDecoder, self).__init__()
        self.decoder_type = decoder_type
        self.decoder_activation = Act(decoder_activation_str)
        self.output_dim = output_dim

        self.decoder: nn.Sequential
        if self.decoder_type == "linear":
            self.decoder = nn.Sequential(nn.Linear(self.output_dim, 3))
        elif self.decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim),
                nn.BatchNorm1d(self.output_dim),
                self.decoder_activation,
                nn.Linear(self.output_dim, 3),
            )
        else:
            raise ValueError(f"Undefined force decoder: {self.decoder_type}")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, x):
        return self.decoder(x)


class InteractionBlock(MessagePassing):
    def __init__(
        self,
        hidden_channels: int,
        mlp_basis_dim: int,
        basis_type,
        depth_mlp_edge: int = 2,
        depth_mlp_trans: int = 1,
        activation_str: str = "ssp",
        ablation: str = "none",
    ) -> None:
        super(InteractionBlock, self).__init__(aggr="add")

        self.activation = Act(activation_str)
        self.ablation = ablation
        self.basis_type = basis_type

        # basis function assumes input is in the range of [-1,1]
        if self.basis_type != "rawcat":
            self.lin_basis = torch.nn.Linear(mlp_basis_dim, hidden_channels)

        if self.ablation == "nocond":
            # the edge filter only depends on edge_attr
            in_features = (
                mlp_basis_dim
                if self.basis_type == "rawcat"
                else hidden_channels
            )
        else:
            # edge filter depends on edge_attr and current node embedding
            in_features = (
                mlp_basis_dim + 2 * hidden_channels
                if self.basis_type == "rawcat"
                else 3 * hidden_channels
            )

        if depth_mlp_edge > 0:
            ### Formula 3, F_e = mlp_edge
            mlp_edge = [torch.nn.Linear(in_features, hidden_channels)]
            for _ in range(depth_mlp_edge):
                mlp_edge.append(self.activation)
                mlp_edge.append(
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
        else:
            ## need batch normalization afterwards. Otherwise training is unstable.
            mlp_edge = [
                torch.nn.Linear(in_features, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]
        self.mlp_edge = torch.nn.Sequential(*mlp_edge)

        if not self.ablation == "nofilter":
            self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        if depth_mlp_trans > 0:
            mlp_trans = [torch.nn.Linear(hidden_channels, hidden_channels)]
            for _ in range(depth_mlp_trans):
                mlp_trans.append(torch.nn.BatchNorm1d(hidden_channels))
                mlp_trans.append(self.activation)
                mlp_trans.append(
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
        else:
            # need batch normalization afterwards. Otherwise, becomes NaN
            mlp_trans = [
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
            ]

        self.mlp_trans = torch.nn.Sequential(*mlp_trans)

        if not self.ablation == "noself":
            self.center_W = torch.nn.Parameter(
                torch.Tensor(1, hidden_channels)
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.basis_type != "rawcat":
            torch.nn.init.xavier_uniform_(self.lin_basis.weight)
            self.lin_basis.bias.data.fill_(0)

        for m in self.mlp_trans:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        for m in self.mlp_edge:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        if not self.ablation == "nofilter":
            torch.nn.init.xavier_uniform_(self.lin.weight)
            self.lin.bias.data.fill_(0)

        if not self.ablation == "noself":
            torch.nn.init.xavier_uniform_(self.center_W)

    def forward(self, x, edge_index, edge_attr, edge_weight):
        if self.basis_type != "rawcat":
            edge_emb = self.lin_basis(edge_attr)
        else:
            # for rawcat, we directly use the raw feature
            edge_emb = edge_attr

        if self.ablation == "nocond":
            emb = edge_emb
        else:
            emb = torch.cat(
                [edge_emb, x[edge_index[0]], x[edge_index[1]]], dim=1
            )

        W = self.mlp_edge(emb) * edge_weight.view(-1, 1)
        if self.ablation == "nofilter":
            x = self.propagate(edge_index, x=x, W=W) + self.center_W
        else:
            x = self.lin(x)
            if self.ablation == "noself":
                x = self.propagate(edge_index, x=x, W=W)
            else:
                x = self.propagate(edge_index, x=x, W=W) + self.center_W * x
        x = self.mlp_trans(x)

        return x

    def message(self, x_j, W):
        if self.ablation == "nofilter":
            return W
        else:
            return x_j * W

class ForceNet(BaseModel):
    r"""Implementation of ForceNet architecture.

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Unused argumebt
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`512`)
        num_iteractions (int, optional): Number of interaction blocks.
            (default: :obj:`5`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        feat (str, optional): Input features to be used
            (default: :obj:`full`)
        num_freqs (int, optional): Number of frequencies for basis function.
            (default: :obj:`50`)
        max_n (int, optional): Maximum order of spherical harmonics.
            (default: :obj:`6`)
        basis (str, optional): Basis function to be used.
            (default: :obj:`full`)
        depth_mlp_edge (int, optional): Depth of MLP for edges in interaction blocks.
            (default: :obj:`2`)
        depth_mlp_node (int, optional): Depth of MLP for nodes in interaction blocks.
            (default: :obj:`1`)
        activation_str (str, optional): Activation function used post linear layer in all message passing MLPs.
            (default: :obj:`swish`)
        ablation (str, optional): Type of ablation to be performed.
            (default: :obj:`none`)
        decoder_hidden_channels (int, optional): Number of hidden channels in the decoder.
            (default: :obj:`512`)
        decoder_type (str, optional): Type of decoder: linear or MLP.
            (default: :obj:`mlp`)
        decoder_activation_str (str, optional): Activation function used post linear layer in decoder.
            (default: :obj:`swinteractionsish`)
        training (bool, optional): If set to :obj:`True`, specify training phase.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,  # not used
        hidden_channels: int = 512,
        num_interactions: int = 5,
        cutoff: float = 6.0,
        feat: str = "full",
        num_freqs: int = 50,
        max_n: int = 3,
        basis: str = "sphallmul",
        depth_mlp_edge: int = 2,
        depth_mlp_node: int = 1,
        activation_str: str = "swish",
        ablation: str = "none",
        decoder_hidden_channels: int = 512,
        decoder_type: str = "mlp",
        decoder_activation_str: str = "swish",
        training: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
        pretrain_return: bool = True,
        **kwargs
    ) -> None:
        super(ForceNet, self).__init__()
        self.training = training
        self.ablation = ablation
        if self.ablation not in [
            "none",
            "nofilter",
            "nocond",
            "nodistlist",
            "onlydist",
            "nodelinear",
            "edgelinear",
            "noself",
        ]:
            raise ValueError(f"Unknown ablation called {ablation}.")

        """
        Descriptions of ablations:
            - none: base ForceNet model
            - nofilter: no element-wise filter parameterization in message modeling
            - nocond: convolutional filter is only conditioned on edge features, not node embeddings
            - nodistlist: no atomic radius information in edge features
            - onlydist: edge features only contains distance information. Orientation information is ommited.
            - nodelinear: node update MLP function is replaced with linear function followed by batch normalization
            - edgelinear: edge MLP transformation function is replaced with linear function followed by batch normalization.
            - noself: no self edge of m_t.
        """

        self.otf_graph = otf_graph
        self.cutoff = cutoff
        self.output_dim = decoder_hidden_channels
        self.feat = feat
        self.num_freqs = num_freqs
        self.num_layers = num_interactions
        self.max_n = max_n
        self.activation_str = activation_str
        self.use_pbc = use_pbc
        self.max_neighbors = 50
        self.pretrain_return = pretrain_return


        # read atom map and atom radii
        atom_map = torch.zeros(101, 9)
        for i in range(101):
            atom_map[i] = torch.tensor(CONTINUOUS_EMBEDDINGS[i])

        atom_radii = torch.zeros(101)
        for i in range(101):
            atom_radii[i] = ATOMIC_RADII[i]
        atom_radii = atom_radii / 100

        self.atom_radii = nn.Parameter(atom_radii, requires_grad=False)
        self.basis_type = basis

        self.pbc_apply_sph_harm = "sph" in self.basis_type
        self.pbc_sph_option = None

        # for spherical harmonics for PBC
        if "sphall" in self.basis_type:
            self.pbc_sph_option = "all"
        elif "sphsine" in self.basis_type:
            self.pbc_sph_option = "sine"
        elif "sphcosine" in self.basis_type:
            self.pbc_sph_option = "cosine"

        self.pbc_sph: Optional[SphericalSmearing] = None
        if self.pbc_apply_sph_harm:
            self.pbc_sph = SphericalSmearing(
                max_n=self.max_n, option=self.pbc_sph_option
            )

        # self.feat can be "simple" or "full"
        if self.feat == "simple":
            self.embedding = nn.Embedding(100, hidden_channels)

            # set up dummy atom_map that only contains atomic_number information
            atom_map = torch.linspace(0, 1, 101).view(-1, 1).repeat(1, 9)
            self.atom_map = nn.Parameter(atom_map, requires_grad=False)

        elif self.feat == "full":
            # Normalize along each dimaension
            atom_map[0] = np.nan
            atom_map_notnan = atom_map[atom_map[:, 0] == atom_map[:, 0]]
            atom_map_min = torch.min(atom_map_notnan, dim=0)[0]
            atom_map_max = torch.max(atom_map_notnan, dim=0)[0]
            atom_map_gap = atom_map_max - atom_map_min

            ## squash to [0,1]
            atom_map = (
                atom_map - atom_map_min.view(1, -1)
            ) / atom_map_gap.view(1, -1)

            self.atom_map = torch.nn.Parameter(atom_map, requires_grad=False)

            in_features = 9
            # first apply basis function and then linear function
            if "sph" in self.basis_type:
                # spherical basis is only meaningful for edge feature, so use powersine instead
                node_basis_type = "powersine"
            else:
                node_basis_type = self.basis_type
            basis = Basis(
                in_features,
                num_freqs=num_freqs,
                basis_type=node_basis_type,
                act=self.activation_str,
            )
            self.embedding = torch.nn.Sequential(
                basis, torch.nn.Linear(basis.out_dim, hidden_channels)
            )

        else:
            raise ValueError("Undefined feature type for atom")

        # process basis function for edge feature
        if self.ablation == "nodistlist":
            # do not consider additional distance edge features
            # normalized (x,y,z) + distance
            in_feature = 4
        elif self.ablation == "onlydist":
            # only consider distance-based edge features
            # ignore normalized (x,y,z)
            in_feature = 4

            # if basis_type is spherical harmonics, then reduce to powersine
            if "sph" in self.basis_type:
                # logging.info(
                #     "Under onlydist ablation, spherical basis is reduced to powersine basis."
                # )
                self.basis_type = "powersine"
                self.pbc_sph = None

        else:
            in_feature = 7

        self.basis_fun = Basis(
            in_feature,
            num_freqs,
            self.basis_type,
            self.activation_str,
            sph=self.pbc_sph,
        )

        # process interaction blocks
        self.interactions = torch.nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels,
                self.basis_fun.out_dim,
                self.basis_type,
                depth_mlp_edge=depth_mlp_edge,
                depth_mlp_trans=depth_mlp_node,
                activation_str=self.activation_str,
                ablation=ablation,
            )
            self.interactions.append(block)

        self.lin = torch.nn.Linear(hidden_channels, self.output_dim)
        self.activation = Act(activation_str)
        self.mask_embedding = nn.Embedding(2, hidden_channels)

        # ForceNet decoder
        self.decoder = FNDecoder(
            decoder_type, decoder_activation_str, self.output_dim
        )
        self.scalar_scale = 1
        self.scalar_offset = 0
        # Projection layer for energy prediction
        self.energy_mlp = nn.Linear(self.output_dim, 1)

    def set_scalar_stat(self, scalar_scale, scalar_offset):
        self.scalar_scale = scalar_scale
        self.scalar_offset = scalar_offset

    def forward(self, 
                z: Tensor,
                pos: Tensor,
                batch: Tensor,
                mask: Optional[Tensor] = None, 
                cell_size: Optional[float] = None,
                regress_force: Optional[bool] = True,
                ):
        z = torch.where(z == 0, torch.tensor(1, dtype=torch.long, device=z.device), torch.tensor(8, dtype=torch.long, device=z.device))
        
        if self.feat == "simple":
            h = self.embedding(z)
        elif self.feat == "full":
            h = self.embedding(self.atom_map[z])
        else:
            raise RuntimeError("Undefined feature type for atom")
        
        ### Generate Mask embeddings
        if mask is None:
            h2 = 0
        else:
            h2 = self.mask_embedding(mask)
        h+=h2
        
        (
            edge_index,
            edge_dist,
            edge_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(pos, batch, cell_size, cutoff=self.cutoff, max_neighbors=self.max_neighbors) #

        if self.pbc_apply_sph_harm:
            edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)
            edge_attr_sph = self.pbc_sph(edge_vec_normalized)

        # calculate the edge weight according to the dist
        edge_weight = torch.cos(0.5 * edge_dist * PI / self.cutoff)

        # normalized edge vectors
        edge_vec_normalized = edge_vec / edge_dist.view(-1, 1)

        # edge distance, taking the atom_radii into account
        # each element lies in [0,1]
        # edge_dist_list is p_sub_st in paper
        edge_dist_list = (
            torch.stack(
                [
                    edge_dist,
                    edge_dist - self.atom_radii[z[edge_index[0]]],
                    edge_dist - self.atom_radii[z[edge_index[1]]],
                    edge_dist
                    - self.atom_radii[z[edge_index[0]]]
                    - self.atom_radii[z[edge_index[1]]],
                ]
            ).transpose(0, 1)
            / self.cutoff
        )

        if self.ablation == "nodistlist":
            edge_dist_list = edge_dist_list[:, 0].view(-1, 1)

        # make sure distance is positive
        edge_dist_list[edge_dist_list < 1e-3] = 1e-3

        # squash to [0,1] for gaussian basis
        if self.basis_type == "gauss":
            edge_vec_normalized = (edge_vec_normalized + 1) / 2.0

        # process raw_edge_attributes to generate edge_attributes
        raw_edge_attr = torch.cat(
            [edge_vec_normalized, edge_dist_list], dim=1
        )

        if "sph" in self.basis_type:
            edge_attr = self.basis_fun(raw_edge_attr, edge_attr_sph)
        else:
            edge_attr = self.basis_fun(raw_edge_attr)

        # pass edge_attributes through interaction blocks
        for _, interaction in enumerate(self.interactions):
            h = h + interaction(h, edge_index, edge_attr, edge_weight)

        h = self.lin(h)
        h_unscaled = self.activation(h)

        h = h_unscaled * self.scalar_scale + self.scalar_offset
        out = scatter(h, batch, dim=0, reduce="add")

        force = self.decoder(h_unscaled) * self.scalar_scale 
        # energy = self.energy_mlp(out)
        if self.pretrain_return:
            return None, force
        else:
            energy = self.energy_mlp(out)
            return energy, force

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())