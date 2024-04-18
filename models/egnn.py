from copy import deepcopy
from numpy import printoptions
from torch import nn
import torch
from torch import Tensor
from torch_cluster import radius_graph
from torch_scatter import scatter
from models.graph_utils import PBC_radius_graph
from typing import Optional, Tuple
import math

class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(
        self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
        residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False,
        use_pbc=False,
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.use_pbc = use_pbc
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        else:
            self.att_mlp = nn.Identity()

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index[0], edge_index[1]
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index[0], edge_index[1]
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0])
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[0], edge_index[1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def coord2radial_pbc(self, edge_index, coord, cell_size, batch):
        # cell_size could be in either shape [b, 3] or a single float
        row, col = edge_index[0], edge_index[1]
        coord_diff = coord[row] - coord[col]   # [b, 3]
        # wrap pbc according to batch
        cell_size_all = torch.zeros((coord_diff.shape[0], 3), device=coord_diff.device)
        cell_size_all[:, 0] = cell_size[batch[edge_index[0]], 0]
        cell_size_all[:, 1] = cell_size[batch[edge_index[0]], 1]
        cell_size_all[:, 2] = cell_size[batch[edge_index[0]], 2]

        coord_diff = torch.remainder(coord_diff + cell_size_all / 2, cell_size_all) - cell_size_all / 2
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, cell_size=None, batch=None):
        row, col = edge_index[0], edge_index[1]
        if self.use_pbc:
            assert cell_size is not None, 'cell_size must be provided when use_pbc is True'
            radial, coord_diff = self.coord2radial_pbc(edge_index, coord, cell_size, batch)
        else:
            radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, 
        hidden_channels, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, 
        residual=True, attention=False, normalize=False, tanh=False, 
        max_atom_type=100, cutoff=5.0, max_num_neighbors=32,
        use_pbc=True, pretrain_return = False, denoise_return = False,
        **kwargs
    ):
        '''
        :param max_atom_type: Number of features for 'h' at the input
        :param hidden_channels: Number of hidden features
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.max_atom_type = max_atom_type
        self.cutoff = cutoff
        self.use_pbc = use_pbc
        self.max_num_neighbors = max_num_neighbors
        self.type_embedding = nn.Embedding(max_atom_type, hidden_channels)
        self.mask_embedding = nn.Embedding(max_atom_type, hidden_channels)
        self.pretrain_return = pretrain_return
        self.denoise_return = denoise_return

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(
                self.hidden_channels, self.hidden_channels, self.hidden_channels, edges_in_d=in_edge_nf,
                act_fn=act_fn, residual=residual, attention=attention, normalize=normalize, tanh=tanh,
                use_pbc=use_pbc))

        self.out_norm = nn.LayerNorm(hidden_channels)
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(), 
            nn.Linear(hidden_channels, 1)
        )

        self.scalar_scale = 1
        self.scalar_offset = 0

    def reset_head(self):
        # reinit energy head after loading pretrained model
        for m in self.energy_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)

    def set_scalar_stat(self, scalar_scale, scalar_offset):
        self.scalar_scale = scalar_scale
        self.scalar_offset = scalar_offset

    def forward(self,
                z: Tensor,
                pos: Tensor,
                batch: Tensor,
                mask: Optional[Tensor] = None,
                edge_index:  Optional[Tensor] = None,
                edge_attr:  Optional[Tensor] = None,
                cell_size: Optional[float] = None,
                regress_force: bool = False) -> Tuple[Tensor, Tensor]:
        
        h1 = self.type_embedding(z)
        if mask is None: ### For MD17, the mask is not provided, thus h2 = 0
            h2 = 0
        else:
            h2 = self.mask_embedding(mask)
        h = h1 + h2
        x = pos
        if regress_force:
            x.requires_grad_(True)
        if edge_index is None:
            if not self.use_pbc:
                edge_index = radius_graph(
                    pos,
                    r=self.cutoff,
                    batch=batch,
                    loop=False,
                    max_num_neighbors=self.max_num_neighbors + 1,
                )
            else:
                assert cell_size is not None, 'cell_size must be provided when use_pbc is True'
                edge_index = PBC_radius_graph(
                    pos,
                    batch,
                    cell_size,
                    self.cutoff,
                    self_interaction=False,
                )
                
        for i in range(0, self.n_layers):
            if self.use_pbc:
                h, x, edge_attr = self._modules["gcl_%d" % i](h, edge_index, x, edge_attr=edge_attr, cell_size=cell_size, batch=batch)
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, edge_index, x, edge_attr=edge_attr)
        h = self.out_norm(h)
        h = self.energy_head(h)

        h = h * self.scalar_scale + self.scalar_offset

        out = scatter(h, batch, dim=0, reduce='add')

        if regress_force:
            force = -1 * (
                torch.autograd.grad(
                    out,
                    pos,
                    grad_outputs=torch.ones_like(out),
                    create_graph=True,
                )[0]
            )
            if self.denoise_return:
                return force, x-pos

            else:
                return out, force # finetune

        return out, x - pos


def unsorted_segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    result_shape = (num_segments, data.shape[1])
    result = torch.zeros(result_shape, device=data.device, dtype=data.dtype)             # data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1,  data.shape[1])
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    result_shape = (num_segments,  data.shape[1])
    segment_ids = segment_ids.unsqueeze(-1).expand(-1,  data.shape[1])
    result =  torch.zeros(result_shape, device=data.device, dtype=data.dtype)                                          # data.new_full(result_shape, 0)  # Init empty result tensor.
    count = torch.zeros(result_shape, device=data.device, dtype=data.dtype)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)