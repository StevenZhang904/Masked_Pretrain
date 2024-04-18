import torch
from torch import Tensor
from torch_nl import compute_neighborlist, compute_neighborlist_n2
from torch_geometric.data import Data
from typing import Optional, Tuple, Union

@torch.no_grad()
def PBC_radius_graph(pos: Tensor,
                     batch: Tensor,
                     cell_size: Union[float, Tensor],
                     cutoff: float,
                     self_interaction: bool=False) -> Tensor:

    if isinstance(cell_size, float):
        cell = torch.eye(3, device=pos.device) * cell_size   # [[cell_size, 0, 0], [0, cell_size, 0], [0, 0, cell_size]]
        cell = cell.repeat(batch.max() + 1, 1)   # just duplicate
    else:   # cell_size can be a tensor in shape [batch_size, 3], where each row corresponds to: [cell_size_x, cell_size_y, cell_size_z]
        cell = torch.eye(3, device=pos.device)

        cell = cell.repeat(batch.max() + 1, 1)   # just duplicate
        cell = cell * cell_size.view(-1, 1).repeat(1, 3)   # multiply each row by cell_size_x, cell_size_y, cell_size_z

    # pbc on all directions
    pbc = torch.tensor([True, True, True], device=pos.device)
    pbc = pbc.repeat(batch.max() + 1)

    # mem = psutil.virtual_memory()
    # print(mem.used/1024/1024/1024)

    # compute neighborlist
    mapping_idx, _, _ = compute_neighborlist(cutoff=cutoff, pos=pos.cpu(),
     cell=cell.cpu(), pbc=pbc.cpu(), batch=batch.cpu(), self_interaction=self_interaction)
    mapping_idx = mapping_idx.to(pos.device)

    # print('=================')
    # mem_new = psutil.virtual_memory()
    # print(mem_new.used/1024/1024/1024)


    return mapping_idx


if __name__ == '__main__':
    # sanity check, compared with numpy kd_tree
    import numpy as np
    from scipy.spatial import cKDTree

    cell_size = 20
    cutoff = 6
    n_atoms = 10
    pos = np.random.rand(n_atoms, 3) * cell_size
    #build kdtree with periodic boundary condition
    tree = cKDTree(pos, boxsize=[cell_size, cell_size, cell_size])
    idx_kdtree = tree.query_ball_point(pos, cutoff)
    
    # process it into geometric style edge_index
    edge_idx_kdtree = []
    for i in range(len(idx_kdtree)):
        for j in idx_kdtree[i]:
            if i != j:
                edge_idx_kdtree.append([i, j])

    pos_tsr = torch.tensor(pos, dtype=torch.float).cuda()
    pos_tsr[:5] -= cell_size
    # data = Data(pos=pos_tsr, batch=torch.zeros(n_atoms, dtype=torch.long).cuda())
    edge_idx = PBC_radius_graph(pos_tsr, torch.zeros(n_atoms, dtype=torch.long).cuda(),
     torch.tensor([cell_size, cell_size, cell_size]).cuda(), cutoff, 
     self_interaction=False)
    print(np.array(edge_idx_kdtree).T)
    print(edge_idx)

    pos_str_wrapped = torch.remainder(pos_tsr, cell_size)
    edge_idx_wrapped = PBC_radius_graph(pos_str_wrapped, torch.zeros(n_atoms, dtype=torch.long).cuda(),
     torch.tensor([cell_size, cell_size, cell_size]).cuda(), cutoff, 
     self_interaction=False)
    print(edge_idx_wrapped)
