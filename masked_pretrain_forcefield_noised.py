from ast import mod
import os
import gc
import yaml
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import wandb

import torch
from torch.optim import AdamW, Adam
import torch.nn.functional as F
import torch.multiprocessing
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

from utils import adjust_learning_rate
from matplotlib import pyplot as plt


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.alternating_task = self.config['alternating_task']
        self.seed = self.config['seed']
        self._seed_everything(self.seed)
        self.loss = self.config['pretrain_loss']

        print('Loss function confirmed ', self.loss)



        if 'water' in self.config['dataset']['data_dir']:
            from dataset.dataset_water import WaterX_noised, WaterX_masked_noised, WaterX_masked
            if self.config['dataset']['masking_ratio'] == 0 and self.config['dataset']['noise_scale']> 0:
                print('Using Noised dataset')
                train_dataset = WaterX_noised(self.config['dataset'], 'train')
                val_dataset = WaterX_noised(self.config['dataset'], 'val')          
            elif self.config['dataset']['noise_scale']> 0 and self.config['dataset']['masking_ratio'] > 0:
                print('Using Masked and Noised dataset')
                train_dataset = WaterX_masked_noised(self.config['dataset'], 'train')
                val_dataset = WaterX_masked_noised(self.config['dataset'], 'val')
            elif self.config['dataset']['masking_ratio'] > 0 and self.config['dataset']['noise_scale'] == 0 or self.config['dataset']['noise_scale'] == None:
                print("Using Masked dataset")
                train_dataset = WaterX_masked(self.config['dataset'], 'train')
                val_dataset = WaterX_masked(self.config['dataset'], 'val')      
            else:     
                raise NotImplementedError     

            self.prefix = 'water'
        elif 'md17' in self.config['dataset']['data_dir']:
            from dataset.dataset_md17 import MD17_masked, file_names, revised_file_names
            if self.config['dataset']['masking_ratio'] > 0 and self.config['dataset']['noise_scale'] == 0 or self.config['dataset']['noise_scale'] == None:
                print('Using Masked dataset for MD17')

                if not self.config['dataset']['single_molecule']:
                    md17_dataconfig = self.config['dataset']
                    dataset_list = revised_file_names if self.config['dataset']['revised'] else file_names
                    all_train_datasets = []
                    all_val_datasets = []
                    for molecule_name in dataset_list:
                        md17_dataconfig['name'] = molecule_name
                        all_train_datasets.append(MD17_masked(md17_dataconfig, mode='train'))
                        all_val_datasets.append(MD17_masked(md17_dataconfig, mode='val'))
                    train_dataset = ConcatDataset(all_train_datasets)
                    val_dataset = ConcatDataset(all_val_datasets)
                    print('Using all datasets')
                else:
                    train_dataset = MD17_masked(self.config['dataset'], mode='train')
                    val_dataset = MD17_masked(self.config['dataset'], mode='val')
        
            self.prefix = 'md17'

        else:
            raise NotImplementedError('Undefined dataset!')
        

        self.train_loader = PyGDataLoader(
                train_dataset, batch_size=self.config['training']['batch_size'], num_workers=self.config['training']['num_workers'], 
                shuffle=True, drop_last=True, 
                pin_memory=True, persistent_workers=True,
                worker_init_fn=lambda k: self._seed_everything(self.seed + (k * 1000)),
            )
        self.val_loader = PyGDataLoader(
                val_dataset, batch_size=500, num_workers=4,
                shuffle=False, drop_last=False,
                pin_memory=True, persistent_workers=True,
                worker_init_fn=lambda k: self._seed_everything(self.seed + (k * 1000)),
            )
        
        if self.config['model']['name'] == 'EGNN':
            self.model_prefix = 'egnn'
        elif self.config['model']['name'] == 'GNS':
            self.model_prefix = 'gns'   
        elif self.config['model']['name'] == 'ForceNet':
            self.model_prefix = 'forcenet'
        else:
            raise NotImplementedError('Undefined model!')
        
        # T dd wandb logger
        wandb.init(project=self.config['project_name'] + '-' + self.model_prefix,
                   tags = ["Pretrain Experiment"],
                   notes = 'After pretrain batch size 16',
                   config=config)
        self.log_dir = self.config['log_dir']+"_"+str(self.config['dataset']['masking_ratio'])+"_"+str(self.config['seed'])
        self._save_config_file(config, self.log_dir)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    @staticmethod
    def _save_config_file(config, ckpt_dir):
        if os.path.exists(ckpt_dir):
            print('Warnings: ckpt_dir exists')
        os.makedirs(ckpt_dir, exist_ok=True)
        # save self.config as config
        with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)


    def _mask_denoise_loss(self,model, data):
        pred_disp, pred_noise = model(data.x, data.pos, data.batch, mask=data.mask, cell_size=data.cell_size,
                        regress_force=True)
        loss_noise = F.mse_loss(
            pred_noise, data.noise, reduction='mean'
        )
        disp = F.normalize(data.disp[data.mask==1], dim=-1)
        print(pred_disp.shape, disp.shape)
        loss_f = 1 - F.cosine_similarity( pred_disp[data.mask==1], disp).mean()
        loss = loss_f + loss_noise
        return loss

    def _denoise_loss(self,model, data):
        _, pred_noise = model(data.x, data.pos, data.batch, cell_size=data.cell_size,
                        regress_force=True)
        loss_noise = F.mse_loss(
            pred_noise, data.noise, reduction='mean'
        )
        return loss_noise
 
    def _masking_loss(self, model, data):
        '''
        Basic assumption here is that we would assume that masking would
        enforce model to learn more comprehensive feature: relative position
        from each other. The underlying mechanism could be enforcing the model to 
        learn from non-energy-minized conformations and thus make it more accurate 
        and robust.
        '''
        if 'cell_size' in data.keys():
            _, pred_disp = model(data.x, data.pos, data.batch, mask=data.mask, cell_size=data.cell_size,
                            regress_force=True)     
            disp = F.normalize(data.disp[data.mask==1], dim=-1)
            loss_f = 1 - F.cosine_similarity( pred_disp[data.mask==1], disp).mean()   
        else:
            _, pred_disp = model(data.x, data.pos, data.batch, regress_force=True)
            disp = F.normalize(data.disp, dim=-1)
            loss_f = 1 - F.cosine_similarity( pred_disp, disp).mean()   

        return loss_f 

    def loss_fn(self, model, data):
        data = data.to(self.device)

        if self.loss == 'denoising':
            loss = self._denoise_loss(model, data)
        elif self.loss == 'masking':
            loss = self._masking_loss(model, data)
        elif self.loss == 'masking and denoising':
            loss = self._mask_denoise_loss(model, data)
        else:
            raise NotImplementedError
        return loss

    def train(self):

        if self.config['model']['name'] == 'EGNN':
            from models.egnn import EGNN         
            model = EGNN(**self.config["model"])
        elif self.config['model']['name'] == 'ForceNet':
            self.model_prefix = 'forcenet'
            from models.forcenet import ForceNet
            model = ForceNet(**self.config["model"])
        elif self.config['model']['name'] == 'GNS':
            self.model_prefix = 'gns'
            from models.learned_simulator import GNS
            model = GNS(**self.config["model"])
        else:
            raise NotImplementedError('Undefined model!')
        
        self._load_weights(model)
        model = model.to(self.device)
        
        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay']) 
        optimizer = Adam(
            model.parameters(), self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        # make sure exists
        os.makedirs(ckpt_dir, exist_ok=True)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):

            # validate the model 
            valid_loss = self._validate(model)
            print('Validation', epoch_counter, 'valid loss', valid_loss)
            wandb.log({'valid_loss': valid_loss})

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))

            valid_n_iter += 1

            with tqdm(range(len(self.train_loader))) as pbar:
                for bn, data in enumerate(self.train_loader):                
                    adjust_learning_rate(optimizer, epoch_counter + bn / len(self.train_loader), self.config)
                    loss = self.loss_fn(model, data)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(1)
                    # set description
                    wandb.log({'train_loss': loss.item()})
                    wandb.log({'lr': optimizer.param_groups[0]['lr']})


                    n_iter += 1
                    pbar.set_description(f'epoch: {epoch_counter}, loss: {loss.item()}')
                gc.collect() # free memory
                torch.cuda.empty_cache()
        # validate the model 
        valid_loss = self._validate(model)
        print('Validation', epoch_counter, 'valid loss', valid_loss)
        wandb.log({'valid_loss': valid_loss})

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))          
            print("model has been saved to ", os.path.join(ckpt_dir, 'model.pth'))
        valid_n_iter += 1
        wandb.finish()

    def _load_weights(self, model):
        try:
            state_dict = torch.load(os.path.join(self.config['load_model'], 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model):
        valid_loss = 0
        model.eval()

        for bn, data in enumerate(self.val_loader):                
            loss = self.loss_fn(model, data)
            valid_loss += loss.item()
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        model.train()
        return valid_loss / (bn+1)

    def _seed_everything(self, seed):
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        seed = (rank * 100000) + seed

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    config = yaml.load(open("config_water_pretrain.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()