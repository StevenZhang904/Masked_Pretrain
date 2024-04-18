from crypt import mksalt
from hmac import new
from math import e
import os
from re import M
from turtle import pos
import numpy as np
from sympy import Q
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.datasets import MD17
from tqdm import tqdm
import random
from torch.utils.data import ConcatDataset

# def save_splits():
#     indices = list(range(100000))
#     random.shuffle(indices)
#     train_indices = indices[:50000]
#     val_indices = indices[50000:51000]
#     test_indices = indices[51000:]

#     ### Write train_indices to a txt file and read it out
#     np.savetxt('md17_train_indices.txt', train_indices, fmt='%d')
#     np.savetxt('md17_val_indices.txt', val_indices, fmt='%d')
#     np.savetxt('md17_test_indices.txt', test_indices, fmt='%d')
#     ### Read the txt file
#     train_indices = np.loadtxt('md17_train_indices.txt', dtype=int)
#     val_indices = np.loadtxt('md17_val_indices.txt', dtype=int)
#     test_indices = np.loadtxt('md17_test_indices.txt', dtype=int)
#     print(len(train_indices), len(val_indices), len(test_indices))
#     assert len(train_indices) + len(val_indices) + len(test_indices) == 100000

file_names = [
    'benzene',
    'uracil',
    'naphtalene',
    'aspirin',
    'salicylic acid',
    'malonaldehyde',
    'ethanol',
    'toluene',
    'paracetamol',
    'azobenzene',
    'revised benzene',
    'revised uracil',
    'revised naphthalene',
    'revised aspirin',
    'revised salicylic acid',
    'revised malonaldehyde',
    'revised ethanol',
    'revised toluene',
    'revised paracetamol',
    'revised azobenzene',
    'benzene CCSD(T)',
    'aspirin CCSD',
    'malonaldehyde CCSD(T)',
    'ethanol CCSD(T)',
    'toluene CCSD(T)',
    'benzene FHI-aims',
]

revised_file_names = [
    'revised benzene',
    'revised uracil',
    'revised naphthalene',
    'revised aspirin',
    'revised salicylic acid',
    'revised malonaldehyde',
    'revised ethanol',
    'revised toluene',
    'revised paracetamol',
    'revised azobenzene']

class MD17_masked(MD17):
    # support DFT dataset with dynamic box sizes
    def __init__(self,
                config,
                mode,
                ):
            
            root = config['root']
            name = config['name']
            masking_ratio = config['masking_ratio']
            pretrain = config['pretrain']
            num_samples = config['num_samples']


            super().__init__(root, name, None)


            self.masking_ratio = masking_ratio
            self.dataset_name = name
            self.num_samples = num_samples

            assert mode in ['train', 'val', 'test'], 'mode should be either train, val, or test'
            self.mode = mode
            self.pretrain = pretrain     

            self._atom_type = super().__getitem__(0).z
            assert 1 in self._atom_type, 'no hydrogen atom in the molecule'


            ### 50000 samples for training, 1000 samples for validation, 49000 samples for testing, according to schnet paper
            if self.dataset_name == 'revised azobenzene':
                train_idx = np.loadtxt('dataset/md17_indices/Revised_Azobenzene_train_indices.txt', dtype=int) 
                val_idx = np.loadtxt('dataset/md17_indices/Revised_Azobenzene_val_indices.txt', dtype=int)
                test_idx = np.loadtxt('dataset/md17_indices/Revised_Azobenzene_test_indices.txt', dtype=int)                
            elif 'revised' in self.dataset_name:
                train_idx = np.loadtxt('dataset/md17_indices/md17_train_indices.txt', dtype=int) 
                val_idx = np.loadtxt('dataset/md17_indices/md17_val_indices.txt', dtype=int)
                test_idx = np.loadtxt('dataset/md17_indices/md17_test_indices.txt', dtype=int)
            else:
                train_idx = np.loadtxt('dataset/md17_indices/{}_50000_train_indices.txt'.format(self.dataset_name), dtype=int) 
                val_idx = np.loadtxt('dataset/md17_indices/{}_50000_val_indices.txt'.format(self.dataset_name), dtype=int)
                test_idx = np.loadtxt('dataset/md17_indices/{}_50000_test_indices.txt'.format(self.dataset_name), dtype=int)                
                print('Using original MD17 dataset')

            assert self.num_samples <= len(train_idx), 'num_samples should be smaller or equal to than the number of samples in the dataset'
            if self.num_samples == -1:
                self.num_samples = len(train_idx)
            if self.mode == 'train':
                self.sample_idx = train_idx[:self.num_samples]
                print('{} samples for training'.format(self.num_samples))
            elif self.mode == 'val':
                self.sample_idx = val_idx
            else:   # test mode
                self.sample_idx = test_idx

    def __len__(self):
        return len(self.sample_idx)

    
    def get_rel_disp(self, pos, mask):
        '''
        Params:
        mask: hydrogen mask from oxygen_positional_encoding
        pos: unmodified positions tensor


        returns: a list of displacements with shape (N - counter, 3)
        '''
        # Corner case: if no atoms are masked out
        if mask.all():
            return torch.zeros_like(pos)
        # Invert mask to get masked out atoms (True for atoms to consider)
        masked_out_atoms = ~mask
                    
        # Get positions of masked out atoms
        masked_out_positions = pos[masked_out_atoms]
        
        # Find indices of non-hydrogen atoms
        # non_hydrogen_atoms = _atom_type != 1
        
        # Get positions of non-hydrogen atoms
        unmasked_positions = pos[mask]             

        displacements = torch.tensor([])
        
        for unmasked_position in unmasked_positions:
            # Calculate displacements between each unmasked atom and the masked out atoms
            displacement = unmasked_position - masked_out_positions
            # print(displacement)
            # Calculate distance between each unmasked atom and the masked out atoms
            distances = torch.linalg.norm(displacement, dim=-1)
            # print(distances)
            # get the largest distance and its index
            max_distance, max_index = torch.max(distances, dim=0)
            # print(max_distance, max_index)
            # Append the displacement with the largest distance to the list
            displacements= torch.cat((displacements, displacement[max_index]), dim=-1)  

        return displacements.reshape(-1, 3)


    def __getitem__(self, idx):
        ### Inhereit from torch_geometric.data.dataset and modify the __getitem__ method
        sample_idx = self.sample_idx[idx]

        if (isinstance(sample_idx, (int, np.integer))
                or (isinstance(sample_idx, Tensor) and sample_idx.dim() == 0)
                or (isinstance(sample_idx, np.ndarray) and np.isscalar(sample_idx))):
            # print('self.indices()', self.indices())
            # print('sample_idx', sample_idx)
            # exit()
            try:
                data = self.get(self.indices()[sample_idx])
            except:
                print(sample_idx, self.sample_idx)
            data = data if self.transform is None else self.transform(data)

            # If pretraining, return the data with hydrogen atoms masked out
            if self.pretrain:
                # Masking hydrogen atoms
                hydrogen_mask = torch.tensor([x != 1 or random.random() >= self.masking_ratio for x in self._atom_type], dtype=torch.bool)
                int_mask = hydrogen_mask.long()
                int_mask = self._atom_type[hydrogen_mask]

                

                ### for each different type of numbers in int_mask, assign a unique number, making sure 1 will also map to another particular number
                # int_mask = torch.unique(int_mask, return_inverse=True)[1]

                # print('hydrogen_mask', int_mask)

                masked_displacements = self.get_rel_disp(data.pos, mask=hydrogen_mask) 
                # print('pos', data.pos)
                # print('masked_displacements', masked_displacements)
                # print('masked_displacements shape', masked_displacements.shape)
                
                ### TODO: x=data.z[hydrogen_mask] would cause errors
                return Data(x=data.z[hydrogen_mask],
                            pos = data.pos[hydrogen_mask],
                            mask = int_mask,
                            disp = masked_displacements, dataset_name=self.dataset_name)
            
            # If not pretraining, return the full data with force and energy
            else:
                return Data(x=data.z, pos=data.pos, e = data.energy, f = data.force)


        else:
            raise NotImplementedError
            return self.index_select(sample_idx)

if __name__ == "__main__":
 
    # Define the root directory where the dataset will be stored
    root_dir = './md17_datasets'

    # Specify the dataset you want to load (e.g., 'benzene')
    # dataset_name = 'revised uracil'

    for dataset_name in ['revised azobenzene']:
        new_dataset = MD17_masked(config={'root': root_dir, 'name': dataset_name, 'masking_ratio': 0.5, 'pretrain': False, 'num_samples': -1}, mode='train')
        print(len(new_dataset))
        print(new_dataset.indices())
        print("sadsa",new_dataset[14880].f)


    # train_loader = PyGDataLoader(
    #         new_dataset, batch_size=4, shuffle=False
    #     )
    # for batch in train_loader:
    #     print(batch.x)
    #     print(batch.pos)
    #     print(batch.pos.shape)
    #     print(batch.mask)
    #     print(batch.disp)
    #     print(batch.disp.shape)
    #     break
    # print("*"   * 50)

    # from dataset_water import WaterX_masked
    # config = {
    #     'data_dir': 'RPBE-waterdataX.npz',
    #     'name': 'RPBE',
    #     'num_samples': -1,
    #     'checkfile': False,
    #     'masking_ratio': 0.5,
    #     'noise_scale': 1,
    #     'alternating_ratio': 0
    # }
    # water_dataset = WaterX_masked(config=config, mode='train')
    # train_loader_1 = PyGDataLoader(
    #         water_dataset, batch_size=4, shuffle=False
    #     )
    # for batch in train_loader_1:
    #     print(batch.x)
    #     print(batch.pos)
    #     print(batch.mask)
    #     print(batch.disp[0:10])
    #     break


