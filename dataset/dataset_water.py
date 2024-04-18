import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
import random
# from openmm.unit import *
# posScale = 1*bohr/angstrom
# energyScale = 1#*(hartree/item)/kilocalorie_per_mole   #*kilojoules_per_mole/kilocalorie_per_mole
# # forceScale = 1*(kilojoules_per_mole /nanometer)/(kilojoules_per_mole /angstrom)
# forceScale = 1*(hartree/item/bohr)/(hartree/item/angstrom)

RPBE_posScale = 0.529177210903
RPBE_energyScale = 1
RPBE_forceScale = 1.8897261246257702

tip3p_posScale = 1
tip3p_energyScale = 1
# tip3p_forceScale = 1*(kilojoules_per_mole / nanometer)/(kilojoules_per_mole /angstrom)
tip3p_forceScale = 0.1
ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
# print('Using following unit scale:')
# print('posScale: ', posScale)
# print('energyScale: ', energyScale)
# print('forceScale: ', forceScale)

class tip3p:
    def __init__(self, m_num, seed_idx, sample_idx, posScale, forceScale, energyScale) -> None:
        self.seed_idx = seed_idx
        self.sample_idx = sample_idx
        self.m_num = m_num
        self.posScale = posScale
        self.forceScale = forceScale
        self.energyScale = energyScale
        self.cell_size = torch.tensor([20,20,20], dtype= torch.float).view(1, 3)
        
        particle_type = []
        for i in range(self.m_num * 3):
            particle_type.append(21 if i % 3 == 0 else 10)  # 21: O, 10: H
        self.particle_type = np.array(particle_type).astype(np.int64).reshape(-1, 1)

    def getitem_tip3p(self, data_dir, idx):
        seed_idx = self.seed_idx[idx // len(self.sample_idx)]
        sample_idx = self.sample_idx[idx % len(self.sample_idx)]
        fname = f'data_{seed_idx}_{sample_idx}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        data_path = os.path.join(data_dir, fname)

        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = torch.tensor(raw_data['pos'], dtype=torch.float).view(-1, 3) * self.posScale
            x = torch.tensor(self.particle_type, dtype=torch.long).view(-1)    # atom type
            assert x.shape[0] == pos.shape[0]
            e = torch.tensor(raw_data['energy'], dtype=torch.float).view(1,-1) * self.energyScale
            f = torch.tensor(raw_data['forces'] , dtype=torch.float).view(-1, 3) * self.forceScale
            pos = torch.remainder(pos, self.cell_size)
            num_atoms = pos.shape[0]
        return x, pos, e, f, self.cell_size, num_atoms

class Water(Dataset):
    def __init__(self,
                 config,
                 mode
                ):
        self.data_dir = config['data_dir']
        self.num_samples = config['num_samples']
        self.num_seeds = config['num_seeds']
        self.checkfile = config['checkfile']
        self.m_num = config['m_num']
        self.add_noise = config['add_noise']
        self.cell_size = config['cell_size']
        if self.add_noise:
            self.noise_std = config['noise_std']
        self.mode = mode


        seed_idx = list(range(self.num_seeds))
        sample_idx = list(range(self.num_samples))
        self.mode = mode
        if self.mode == 'train':
            self.seed_idx = seed_idx[:int(self.num_seeds*0.8)]
            self.sample_idx = sample_idx
        elif self.mode == 'val':
            self.seed_idx = seed_idx[int(self.num_seeds*0.8):int(self.num_seeds*0.9)]
            self.sample_idx = sample_idx
        else:   # test mode
            self.seed_idx = seed_idx[int(self.num_seeds*0.9):]
            self.sample_idx = sample_idx

        if self.checkfile:
            self._checkfile()
        
        particle_type = np.array([])
        for i in range(self.m_num * 3):
            particle_type.append(21 if i % 3 == 0 else 10)  # 21: O, 10: H
        self.particle_type = np.array(particle_type).astype(np.int64).reshape(-1, 1)

    def __len__(self):
        return len(self.seed_idx) * len(self.sample_idx)

    def _checkfile(self):
        # loop through all the files see if they are all there
        for seed in tqdm(self.seed_idx):
            for sample in self.sample_idx:
                filename = os.path.join(self.data_dir, f'data_{seed}_{sample}.npz')
                try:
                    np.load(filename)
                except:
                    print(f'{filename} not found')
                    return False  

    def __getitem__(self, idx):
        # fetch seed and sample index
        seed_idx = self.seed_idx[idx // len(self.sample_idx)]
        sample_idx = self.sample_idx[idx % len(self.sample_idx)]

        fname = f'data_{seed_idx}_{sample_idx}'#f'seed_{seed_to_read}_data_{sample_to_read}'
        data_path = os.path.join(self.data_dir, fname)

        with np.load(data_path + '.npz', 'rb') as raw_data:
            pos = torch.tensor(raw_data['pos'], dtype=torch.float).view(-1, 3) * tip3p_posScale
            x = torch.tensor(self.particle_type, dtype=torch.long).view(-1)    # atom type
            assert x.shape[0] == pos.shape[0]
            e = torch.tensor(raw_data['energy'], dtype=torch.float).view(1,-1) * tip3p_energyScale
            f = torch.tensor(raw_data['forces'] , dtype=torch.float).view(-1, 3) * tip3p_forceScale
            if self.add_noise:
                noise = torch.randn_like(pos) * self.noise_std
                corrupted_pos = pos + noise
                # wrap position back to box
                cell_size = self.cell_size
                corrupted_pos = corrupted_pos % cell_size
                data = Data(x=x, pos=corrupted_pos, noise=noise, e=e, f=f)
            else:
                data = Data(x=x, pos=pos, e=e, f=f)


        return data

class WaterX(Dataset):
    # support DFT dataset with dynamic box sizes
    def __init__(self,
                 config,
                 mode
                ):
        self.data_dir = config['data_dir']
        self.num_samples = config['num_samples']
        self.checkfile = config['checkfile']
        self.dataset_name = config['name']
        # self.add_noise = config['add_noise']
        # if self.add_noise:
        #     self.noise_std = config['noise_std']
        self.mode = mode

        if self.dataset_name == 'RPBE':
            with np.load(self.data_dir, allow_pickle=True) as npz_data:
                train_idx = npz_data['train_idx']
                test_idx = npz_data['test_idx']
                self.pos = npz_data['pos']
                self.energies = npz_data['energy']
                self.forces = npz_data['force']
                self.box_size = npz_data['box']
                self.atom_type = npz_data['atom_type']

            assert self.num_samples <= len(train_idx), 'num_samples should be smaller than the number of samples in the dataset'
            if self.num_samples == -1:
                self.num_samples = len(train_idx)
            if self.mode == 'train':
                self.sample_idx = train_idx[:self.num_samples][:int(len(train_idx)*0.9)]
            elif self.mode == 'val':
                self.sample_idx = train_idx[:self.num_samples][int(len(train_idx)*0.9):]
            else:   # test mode
                self.sample_idx = test_idx

            self.posScale = RPBE_posScale
            self.energyScale = RPBE_energyScale
            self.forceScale = RPBE_forceScale


        elif self.dataset_name == 'tip3p':
            self.m_num = config['m_num']
            self.num_seeds = config['num_seeds']

            seed_idx = list(range(self.num_seeds))
            sample_idx = list(range(self.num_samples))
            self.mode = mode
            if self.mode == 'train':
                self.seed_idx = seed_idx[:int(self.num_seeds*0.8)]
                self.sample_idx = sample_idx
            elif self.mode == 'val':
                self.seed_idx = seed_idx[int(self.num_seeds*0.8):int(self.num_seeds*0.9)]
                self.sample_idx = sample_idx
            else:   # test mode
                self.seed_idx = seed_idx[int(self.num_seeds*0.9):]
                self.sample_idx = sample_idx

            self.posScale = tip3p_posScale
            self.energyScale = tip3p_energyScale
            self.forceScale = tip3p_forceScale     

            self.tip3p = tip3p(m_num=self.m_num,
                               seed_idx=self.seed_idx,
                               sample_idx=self.sample_idx,
                               posScale=self.posScale,
                               forceScale=self.forceScale,
                               energyScale=self.energyScale)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.dataset_name == 'tip3p':
            return len(self.seed_idx) * len(self.sample_idx)
        elif self.dataset_name == 'RPBE':
            return len(self.sample_idx)

    def __getitem__(self, idx):
        if self.dataset_name == 'RPBE':
            # fetch seed and sample index
            sample_idx = self.sample_idx[idx]

            pos = torch.tensor(self.pos[sample_idx].copy(), dtype=torch.float).view(-1, 3) * self.posScale

            atom_type_raw = self.atom_type[sample_idx].copy()
            # map 0/1 to 10/21
            atom_type = []
            for i in range(len(atom_type_raw)):
                atom_type.append(21 if atom_type_raw[i] == 1 else 10)
            x = torch.tensor(atom_type, dtype=torch.long).view(-1)    # atom type
            assert x.shape[0] == pos.shape[0]
            e = torch.tensor(self.energies[sample_idx].copy(), dtype=torch.float).view(1,-1) * self.energyScale
            f = torch.tensor(self.forces[sample_idx].copy(), dtype=torch.float).view(-1, 3) * self.forceScale
            cell_size = torch.tensor(self.box_size[sample_idx].copy(), dtype=torch.float).view(1, 3) * self.posScale
            
            pos = torch.remainder(pos, cell_size)
            num_atoms = pos.shape[0]
        # ensure positions are within box

        elif self.dataset_name =='tip3p':       
            x, pos, e, f, cell_size, num_atoms = self.tip3p.getitem_tip3p(data_dir = self.data_dir, idx=idx)
        # if self.add_noise:
        #     noise = torch.randn_like(pos) * self.noise_std
        #     corrupted_pos = pos + noise
        #     # wrap position back to box
        #     corrupted_pos = torch.remainder(corrupted_pos, cell_size)
        #     data = Data(x=x, pos=corrupted_pos, noise=noise, e=e, f=f, cell_size=cell_size)
        # else:
        data = Data(x=x, pos=pos, e=e, f=f,
                    cell_size=cell_size, n=num_atoms)

        return data
    
class WaterX_noised(Dataset):
    # support DFT dataset with dynamic box sizes
    def __init__(self,
                 config,
                 mode
                ):
        self.data_dir = config['data_dir']
        self.num_samples = config['num_samples']
        self.checkfile = config['checkfile']
        self.noise_scale = config['noise_scale']
        self.mode = mode
        self.dataset_name = config['name']

        if self.dataset_name == 'RPBE':
            with np.load(self.data_dir, allow_pickle=True) as npz_data:
                train_idx = npz_data['train_idx']
                test_idx = npz_data['test_idx']
                self.pos = npz_data['pos']
                self.energies = npz_data['energy']
                self.forces = npz_data['force']
                self.box_size = npz_data['box']
                self.atom_type = npz_data['atom_type']

            assert self.num_samples <= len(train_idx), 'num_samples should be smaller than the number of samples in the dataset'
            if self.num_samples == -1:
                self.num_samples = len(train_idx)
            if self.mode == 'train':
                self.sample_idx = train_idx[:self.num_samples][:int(len(train_idx)*0.9)]
            elif self.mode == 'val':
                self.sample_idx = train_idx[:self.num_samples][int(len(train_idx)*0.9):]
            else:   # test mode
                self.sample_idx = test_idx

            self.posScale = RPBE_posScale
            self.energyScale = RPBE_energyScale
            self.forceScale = RPBE_forceScale


        elif self.dataset_name == 'tip3p':
            self.m_num = config['m_num']
            self.num_seeds = config['num_seeds']

            seed_idx = list(range(self.num_seeds))
            sample_idx = list(range(self.num_samples))
            self.mode = mode
            if self.mode == 'train':
                self.seed_idx = seed_idx[:int(self.num_seeds*0.8)]
                self.sample_idx = sample_idx
            elif self.mode == 'val':
                self.seed_idx = seed_idx[int(self.num_seeds*0.8):int(self.num_seeds*0.9)]
                self.sample_idx = sample_idx
            else:   # test mode
                self.seed_idx = seed_idx[int(self.num_seeds*0.9):]
                self.sample_idx = sample_idx

            self.posScale = tip3p_posScale
            self.energyScale = tip3p_energyScale
            self.forceScale = tip3p_forceScale     

            self.tip3p = tip3p(m_num=self.m_num,
                               seed_idx=self.seed_idx,
                               sample_idx=self.sample_idx,
                               posScale=self.posScale,
                               forceScale=self.forceScale,
                               energyScale=self.energyScale)
        else:
            raise NotImplementedError   


    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, idx):
        # fetch seed and sample index
        sample_idx = self.sample_idx[idx]

        pos = torch.tensor(self.pos[sample_idx].copy(), dtype=torch.float).view(-1, 3) * self.posScale

        atom_type_raw = self.atom_type[sample_idx].copy()
        # map 0/1 to 10/21
        atom_type = []
        for i in range(len(atom_type_raw)):
            atom_type.append(21 if atom_type_raw[i] == 1 else 10)
        x = torch.tensor(atom_type, dtype=torch.long).view(-1)    # atom type
        assert x.shape[0] == pos.shape[0]
        # e = torch.tensor(self.energies[sample_idx].copy(), dtype=torch.float).view(1,-1) * self.energyScale
        # f = torch.tensor(self.forces[sample_idx].copy(), dtype=torch.float).view(-1, 3) * self.forceScale
        cell_size = torch.tensor(self.box_size[sample_idx].copy(), dtype=torch.float).view(1, 3) * self.posScale
        # ensure positions are within box
        pos = torch.remainder(pos, cell_size)
        num_atoms = pos.shape[0]

        noise = torch.randn_like(pos) * self.noise_scale
        corrupted_pos = pos + noise
        # wrap position back to box
        corrupted_pos = torch.remainder(corrupted_pos, cell_size)
        data = Data(x=x, pos=corrupted_pos, noise=noise, cell_size=cell_size, n=num_atoms)
        return data


class WaterX_masked(Dataset):
    # support DFT dataset with dynamic box sizes
    def __init__(self,
                 config,
                 mode
                ):
        self.data_dir = config['data_dir']
        self.num_samples = config['num_samples']
        self.checkfile = config['checkfile']
        self.masking_ratio = config['masking_ratio']
        self.mode = mode
        self.dataset_name = config['name']

        if self.dataset_name == 'RPBE':
            with np.load(self.data_dir, allow_pickle=True) as npz_data:
                train_idx = npz_data['train_idx']
                test_idx = npz_data['test_idx']
                self.pos = npz_data['pos']
                self.energies = npz_data['energy']
                self.forces = npz_data['force']
                self.box_size = npz_data['box']
                self.atom_type = npz_data['atom_type']

            assert self.num_samples <= len(train_idx), 'num_samples should be smaller than the number of samples in the dataset'
            if self.num_samples == -1:
                self.num_samples = len(train_idx)
            if self.mode == 'train':
                self.sample_idx = train_idx[:self.num_samples][:int(len(train_idx)*0.9)]
            elif self.mode == 'val':
                self.sample_idx = train_idx[:self.num_samples][int(len(train_idx)*0.9):]
            else:   # test mode
                self.sample_idx = test_idx

            self.posScale = RPBE_posScale
            self.energyScale = RPBE_energyScale
            self.forceScale = RPBE_forceScale

        elif self.dataset_name == 'tip3p':
            self.m_num = config['m_num']
            self.num_seeds = config['num_seeds']

            seed_idx = list(range(self.num_seeds))
            sample_idx = list(range(self.num_samples))
            self.mode = mode
            if self.mode == 'train':
                self.seed_idx = seed_idx[:int(self.num_seeds*0.8)]
                self.sample_idx = sample_idx
            elif self.mode == 'val':
                self.seed_idx = seed_idx[int(self.num_seeds*0.8):int(self.num_seeds*0.9)]
                self.sample_idx = sample_idx
            else:   # test mode
                self.seed_idx = seed_idx[int(self.num_seeds*0.9):]
                self.sample_idx = sample_idx

            self.posScale = tip3p_posScale
            self.energyScale = tip3p_energyScale
            self.forceScale = tip3p_forceScale     

            self.tip3p = tip3p(m_num=self.m_num,
                               seed_idx=self.seed_idx,
                               sample_idx=self.sample_idx,
                               posScale=self.posScale,
                               forceScale=self.forceScale,
                               energyScale=self.energyScale)
            # cell_size =  torch.tensor([20,20,20], dtype= torch.float32).reshape(-1,1)
            # self.box_size = cell_size.unsqueeze(0).expand(len(train_idx)+len(test_idx), -1)
        else:
            raise NotImplementedError
        
    def create_mask(self, atom_type, ratio):
        '''
        returns a mask that mask out hydrogen positions w.r.t their connected oxygen atoms,
        which are selected with given ratio 
        mask_value: 
        0: nothing
        1: in a molecule whose hydrogen has been masked
        2: is the hydrogen atom that has been masked
        '''
        num_molecules = len(atom_type) // 3
        # randomly select molecules according to ratio
        # print('num_molecules: ', num_molecules)
        num_selected = int(num_molecules * ratio)
        assert num_selected > 0, 'masking ratio is too small'
        selected_idx = np.random.choice(num_molecules, num_selected, replace=False)
        # create mask
        mask = torch.zeros_like(atom_type, dtype=torch.long)
        for i in selected_idx:
            # Here we assume that data is in repetition of (O,H,H) sequence
            mask[i*3] = 1
            if np.random.rand() < 0.5:  # break tie
                mask[i*3+1] = 1
                mask[i*3+2] = 2
            else:
                mask[i*3+2] = 1
                mask[i*3+1] = 2
        return mask

    
    def get_rel_disp(self, mask, pos, cell_size):
        '''
        Params:
        mask: hydrogen mask from oxygen_positional_encoding
        pos: unmodified positions tensor
        counter: num of masked hydrogen atoms from oxygen_positional_encoding

        Among all water molecules, using masks to identify which water mol has been selected
        by hydrogen mask, for that particular water mol, calculate the displacements between
        the unmasked atoms to the masked hydrogen atom. 

        returns: a list of displacements with shape (N - counter, 3)
        '''
        masked_disp = torch.zeros_like(pos)
        for i in range(0, len(mask), 3):
            if mask[i] == 1:
                for k in range(3):
                    if mask[i+2] == 2: 
                        masked_disp[i][k] = pos[i+2][k] - pos[i][k]
                        masked_disp[i+1][k] = pos[i+2][k] - pos[i+1][k]
                    elif mask[i+1] == 2:
                        masked_disp[i][k] = pos[i+1][k] - pos[i][k]
                        masked_disp[i+2][k] = pos[i+1][k] - pos[i+2][k]

                # fist, make sure the displacement is within the box
                masked_disp = torch.remainder(masked_disp + cell_size/2., cell_size) - cell_size/2.
                # normalize
                masked_disp[i] = masked_disp[i] / torch.norm(masked_disp[i])
                masked_disp[i+1] = masked_disp[i+1] / torch.norm(masked_disp[i+1])

        masked_disp = masked_disp[mask != 2].view(-1,3) # remove masked hydrogen from it
        return masked_disp     


    def __len__(self):
        if self.dataset_name == 'tip3p':
            return len(self.seed_idx) * len(self.sample_idx)
        elif self.dataset_name == 'RPBE':
            return len(self.sample_idx)
        
    def __getitem__(self, idx):
        if self.dataset_name == 'RPBE':

            # fetch seed and sample index
            sample_idx = self.sample_idx[idx]

            pos = torch.tensor(self.pos[sample_idx], dtype=torch.float).view(-1, 3) * self.posScale

            atom_type_raw = self.atom_type[sample_idx]
            # map 0/1 to 10/21
            # 21: oxygen/ 10: hydrogen
            atom_type = []
            for i in range(len(atom_type_raw)):
                atom_type.append(21 if atom_type_raw[i] == 1 else 10)
            assert len(atom_type)%3 == 0 
            x = torch.tensor(atom_type, dtype=torch.long).view(-1)    # atom type
            # e = torch.tensor(self.energies[sample_idx], dtype=torch.float).view(1,-1) * energyScale
            # f = torch.tensor(self.forces[sample_idx], dtype=torch.float).view(-1, 3) * forceScale
            cell_size = torch.tensor(self.box_size[sample_idx], dtype=torch.float).view(1, 3) * self.posScale
            # ensure positions are within box
            pos = torch.remainder(pos, cell_size)

        elif self.dataset_name =='tip3p':       
            x, pos, e, f, cell_size, num_atoms = self.tip3p.getitem_tip3p(data_dir = self.data_dir, idx=idx)
        
        # if self.add_noise:
        #     noise = torch.randn_like(pos) * self.noise_std

        #     if self.hydrogen_mask:
        #         masked_noise = self.noise_mask(noise, atom_type)
        #         corrupted_pos = pos + masked_noise
        #         corrupted_pos = torch.remainder(corrupted_pos, cell_size)
        #         data = Data(x=x, pos=corrupted_pos, noise=noise, e=e, f=f, cell_size=cell_size, masked_noise = masked_noise)
        #     else:
        #         corrupted_pos = pos + noise
        #         # wrap position back to box
        #         corrupted_pos = torch.remainder(corrupted_pos, cell_size)
        #         data = Data(x=x, pos=corrupted_pos, noise=noise, e=e, f=f, cell_size=cell_size)
            
        # else:
        hydrogen_mask = self.create_mask(x, ratio=self.masking_ratio) # generate mask

        masked_disp = self.get_rel_disp(hydrogen_mask, pos, cell_size) # get displacements

        data = Data(x=x[hydrogen_mask!=2], pos=pos[hydrogen_mask!=2].view(-1, 3),  cell_size=cell_size, 
                    mask = hydrogen_mask[hydrogen_mask!=2], 
                    disp = masked_disp)                
        return data
    

class WaterX_masked_noised(Dataset):
    # support DFT dataset with dynamic box sizes
    def __init__(self,
                 config,
                 mode
                ):
        self.data_dir = config['data_dir']
        self.num_samples = config['num_samples']
        self.checkfile = config['checkfile']
        self.masking_ratio = config['masking_ratio']
        self.mode = mode
        self.noise_scale = config['noise_scale']
        self.dataset_name = config['name']
        self.alternating_ratio = False

        if self.dataset_name == 'RPBE':
            with np.load(self.data_dir, allow_pickle=True) as npz_data:
                train_idx = npz_data['train_idx']
                test_idx = npz_data['test_idx']
                self.pos = npz_data['pos']
                self.energies = npz_data['energy']
                self.forces = npz_data['force']
                self.box_size = npz_data['box']
                self.atom_type = npz_data['atom_type']

            assert self.num_samples <= len(train_idx), 'num_samples should be smaller than the number of samples in the dataset'
            if self.num_samples == -1:
                self.num_samples = len(train_idx)
            if self.mode == 'train':
                self.sample_idx = train_idx[:self.num_samples][:int(len(train_idx)*0.9)]
            elif self.mode == 'val':
                self.sample_idx = train_idx[:self.num_samples][int(len(train_idx)*0.9):]
            else:   # test mode
                self.sample_idx = test_idx

            self.posScale = RPBE_posScale
            self.energyScale = RPBE_energyScale
            self.forceScale = RPBE_forceScale

        elif self.dataset_name == 'tip3p':
            self.m_num = config['m_num']
            self.num_seeds = config['num_seeds']

            seed_idx = list(range(self.num_seeds))
            sample_idx = list(range(self.num_samples))
            self.mode = mode
            if self.mode == 'train':
                self.seed_idx = seed_idx[:int(self.num_seeds*0.8)]
                self.sample_idx = sample_idx
            elif self.mode == 'val':
                self.seed_idx = seed_idx[int(self.num_seeds*0.8):int(self.num_seeds*0.9)]
                self.sample_idx = sample_idx
            else:   # test mode
                self.seed_idx = seed_idx[int(self.num_seeds*0.9):]
                self.sample_idx = sample_idx

            self.posScale = tip3p_posScale
            self.energyScale = tip3p_energyScale
            self.forceScale = tip3p_forceScale     

            self.tip3p = tip3p(m_num=self.m_num,
                               seed_idx=self.seed_idx,
                               sample_idx=self.sample_idx,
                               posScale=self.posScale,
                               forceScale=self.forceScale,
                               energyScale=self.energyScale)
            # cell_size =  torch.tensor([20,20,20], dtype= torch.float32).reshape(-1,1)
            # self.box_size = cell_size.unsqueeze(0).expand(len(train_idx)+len(test_idx), -1)
        else:
            raise NotImplementedError
        
    def _create_mask(self, atom_type, ratio):
        '''
        returns a mask that mask out hydrogen positions w.r.t their connected oxygen atoms,
        which are selected with given ratio 
        mask_value: 
        0: nothing
        1: in a molecule whose hydrogen has been masked
        2: is the hydrogen atom that has been masked
        '''
        num_molecules = len(atom_type) // 3
        # randomly select molecules according to ratio
        # print('num_molecules: ', num_molecules)
        num_selected = int(num_molecules * ratio)
        assert num_selected > 0, 'masking ratio is too small'
        selected_idx = np.random.choice(num_molecules, num_selected, replace=False)
        # create mask
        mask = torch.zeros_like(atom_type, dtype=torch.long)
        for i in selected_idx:
            # Here we assume that data is in repetition of (O,H,H) sequence
            mask[i*3] = 1
            if np.random.rand() < 0.5:  # break tie
                mask[i*3+1] = 1
                mask[i*3+2] = 2
            else:
                mask[i*3+2] = 1
                mask[i*3+1] = 2
        return mask

    
    def _get_rel_disp(self, mask, pos, cell_size):
        '''
        Params:
        mask: hydrogen mask from oxygen_positional_encoding
        pos: unmodified positions tensor
        counter: num of masked hydrogen atoms from oxygen_positional_encoding

        Among all water molecules, using masks to identify which water mol has been selected
        by hydrogen mask, for that particular water mol, calculate the displacements between
        the unmasked atoms to the masked hydrogen atom. 

        returns: a list of displacements with shape (N - counter, 3)
        '''
        masked_disp = torch.zeros_like(pos)
        for i in range(0, len(mask), 3):
            if mask[i] == 1:
                for k in range(3):
                    if mask[i+2] == 2: 
                        masked_disp[i][k] = pos[i+2][k] - pos[i][k]
                        masked_disp[i+1][k] = pos[i+2][k] - pos[i+1][k]
                    elif mask[i+1] == 2:
                        masked_disp[i][k] = pos[i+1][k] - pos[i][k]
                        masked_disp[i+2][k] = pos[i+1][k] - pos[i+2][k]

                # fist, make sure the displacement is within the box
                masked_disp = torch.remainder(masked_disp + cell_size/2., cell_size) - cell_size/2.
                # normalize
                masked_disp[i] = masked_disp[i] / torch.norm(masked_disp[i])
                masked_disp[i+1] = masked_disp[i+1] / torch.norm(masked_disp[i+1])

        masked_disp = masked_disp[mask != 2].view(-1,3) # remove masked hydrogen from it
        return masked_disp     


    def __len__(self):
        if self.dataset_name == 'tip3p':
            return len(self.seed_idx) * len(self.sample_idx)
        elif self.dataset_name == 'RPBE':
            return len(self.sample_idx)
        

    def __getitem__(self, idx):
        if self.dataset_name == 'RPBE':

            # fetch seed and sample index
            sample_idx = self.sample_idx[idx]

            pos = torch.tensor(self.pos[sample_idx], dtype=torch.float).view(-1, 3) * self.posScale

            atom_type_raw = self.atom_type[sample_idx]
            # map 0/1 to 10/21
            # 21: oxygen/ 10: hydrogen
            atom_type = []
            for i in range(len(atom_type_raw)):
                atom_type.append(21 if atom_type_raw[i] == 1 else 10)
            assert len(atom_type)%3 == 0 
            x = torch.tensor(atom_type, dtype=torch.long).view(-1)    # atom type
            # e = torch.tensor(self.energies[sample_idx], dtype=torch.float).view(1,-1) * energyScale
            # f = torch.tensor(self.forces[sample_idx], dtype=torch.float).view(-1, 3) * forceScale
            cell_size = torch.tensor(self.box_size[sample_idx], dtype=torch.float).view(1, 3) * self.posScale
            # ensure positions are within box
            pos = torch.remainder(pos, cell_size)

        elif self.dataset_name =='tip3p':       
            x, pos, e, f, cell_size, num_atoms = self.tip3p.getitem_tip3p(data_dir = self.data_dir, idx=idx)
        

        # noise= torch.rand((len(x)//3,1)).reshape(-1,1) * self.config['noise_scale']
        # noise= torch.cat((noise, noise, noise), dim = 1).reshape(-1,1)
        # noise= torch.cat((noise, noise, noise), dim = -1)

        noise = torch.randn_like(pos) * self.noise_scale
        corrupted_pos = pos + noise

        # wrap position back to box
        corrupted_pos = torch.remainder(corrupted_pos, cell_size)

        ### alternating_task: 0 is masking, 1 is denoise + masking task. 
        hydrogen_mask = self._create_mask(x, ratio=self.masking_ratio) # generate mask

        masked_disp = self._get_rel_disp(hydrogen_mask, pos, cell_size) # get displacements
        if self.alternating_ratio:
            alternating_tasks = torch.tensor((0 if np.random.rand() < self.alternating_ratio else 1), dtype=torch.float)
            data = Data(x=x[hydrogen_mask!=2], pos=corrupted_pos[hydrogen_mask!=2].view(-1, 3),  cell_size=cell_size, 
                    mask = hydrogen_mask[hydrogen_mask!=2], 
                    disp = masked_disp,
                    noise = noise[hydrogen_mask!=2],
                    alternating_tasks = alternating_tasks)
        else:
            data = Data(x=x[hydrogen_mask!=2], pos=corrupted_pos[hydrogen_mask!=2].view(-1, 3),  cell_size=cell_size, 
                    mask = hydrogen_mask[hydrogen_mask!=2], 
                    disp = masked_disp,
                    noise = noise[hydrogen_mask!=2])
        return data


if __name__ == "__main__":
    print("X")