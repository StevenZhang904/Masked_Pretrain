from math import ceil
import os
import gc
import time
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
from torch_geometric.loader import DataLoader as PyGDataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

from utils import adjust_learning_rate
from matplotlib import pyplot as plt


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()

        self.seed = self.config['seed']
        self._seed_everything(self.seed)

        if self.config['model']['name'] == 'EGNN':
            self.model_prefix = 'egnn'
        elif self.config['model']['name'] == 'ForceNet':
            self.model_prefix = 'forcenet'
        elif self.config['model']['name'] == 'GNS':
            self.model_prefix = 'gns'            
        else:
            raise NotImplementedError('Undefined model!')
        
        wandb.init(project=self.config['project_name'] + '-' + self.model_prefix,
                    tags = ["Finetune Experiment"],
                    config=config)
        self.log_dir = self.config['log_dir'] + "_" + str(self.config['seed']) + "_" + self.config['dataset']['name']
        self._save_config_file(config, self.log_dir)

        self.e_mean = 0
        self.e_std = 1

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
        with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

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

    def loss_fn(self, model, data):
        
        data = data.to(self.device)

        if 'cell_size' in data.keys():
            pred_e, pred_f = model(data.x, data.pos, data.batch, cell_size=data.cell_size, regress_force=True)

        else:
            pred_e, pred_f = model(data.x, data.pos, data.batch, regress_force=True)

        loss_f = F.l1_loss(
            pred_f * self.config['training']['force_weight'],
            (data.f * self.config['training']['force_weight'])/self.e_std, reduction='mean'
        )
        loss_e = F.l1_loss(
            pred_e.reshape(data.e.shape), (data.e-self.e_mean)/self.e_std, reduction='mean'
        )
    
        loss = loss_e*self.config['training']['energy_weight'] + loss_f
        loss_dict = {
            'energy loss': loss_e.detach(), 
            'force loss': loss_f.detach()
        }
        return pred_e, pred_f, loss, loss_dict

    def _denorm(self, input, e_std, e_mean):
        return input*e_std+e_mean

    def train(self):

        # create train loader
        if 'water' in self.config['dataset']['data_dir']:
            from dataset.dataset_water import WaterX
            train_dataset = WaterX(self.config['dataset'], 'train')
            val_dataset = WaterX(self.config['dataset'], 'val')
            test_dataset = WaterX(self.config['dataset'], 'test')

            train_loader = PyGDataLoader(
                    train_dataset, batch_size=self.config['training']['batch_size'], num_workers=self.config['training']['num_workers'], 
                    shuffle=True, drop_last=True,
                    worker_init_fn=lambda k: self._seed_everything(self.seed + (k * 1000)),
                )
          
            self.prefix = 'water'
        elif 'md17' in self.config['dataset']['data_dir']:
            from dataset.dataset_md17 import MD17_masked

            print('Using Masked dataset for MD17')
            train_dataset = MD17_masked(self.config['dataset'], mode='train')
            val_dataset = MD17_masked(self.config['dataset'], mode='val')
            test_dataset = MD17_masked(self.config['dataset'], mode='test')
            train_loader = PyGDataLoader(
                    train_dataset, batch_size=self.config['training']['batch_size'], num_workers=self.config['training']['num_workers'], 
                    shuffle=True, drop_last=True, 
                    pin_memory=True, persistent_workers=True,
                    worker_init_fn=lambda k: self._seed_everything(self.seed + (k * 1000)),
                )
            print('Training set size:   {}'.format(len(train_dataset)))
            print('Validation set size: {}'.format(len(val_dataset)))
            print('Testing set size:    {}\n'.format(len(test_dataset)))
            self.prefix = 'md17'
        else:
            raise NotImplementedError('Undefined dataset!')

        if self.config['dataset']['name'] == 'RPBE':
            e_labels = []
            atom_counts = []
            for i, d in enumerate(train_loader):
                e_label = d.e.view(-1)
                atom_count = d.n.view(-1)
                e_labels.append(e_label)
                atom_counts.append(atom_count)
                if i % 5000 == 0:
                    print('normalizing', i)
            e_labels = torch.cat(e_labels)
            atom_counts = torch.cat(atom_counts)

            per_atom_e = e_labels / atom_counts
            self.e_mean = per_atom_e.mean()
            self.e_std = per_atom_e.std()
        
        elif self.config['dataset']['name'] == 'tip3p':
            e_labels = []
            atom_counts = []
            for i, d in enumerate(train_loader):
                e_label = d.e.view(-1)
                e_labels.append(e_label)
                if i % 5000 == 0:
                    print('normalizing', i)
            e_labels = torch.cat(e_labels)
            self.e_mean = e_labels.mean()
            self.e_std = e_labels.std()

        elif self.prefix == 'md17':
            e_labels = []
            for i, d in enumerate(train_loader):
                e_label = d.e.view(-1)
                e_labels.append(e_label)
                if i % 5000 == 0:
                    print('normalizing', i)
            e_labels = torch.cat(e_labels)
            self.e_mean = e_labels.mean()
            self.e_std = e_labels.std()

        print(self.e_mean, self.e_std, e_labels.shape)
        wandb.log({'e_mean': self.e_mean, 'e_std': self.e_std})


        del e_labels
        gc.collect() # free memory

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


        model.set_scalar_stat(self.e_std, self.e_mean)
        
        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay']) 
        optimizer = AdamW(
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
            with tqdm(range(len(train_loader))) as pbar:
                for bn, data in enumerate(train_loader):                
                    adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                    __, _, loss, loss_dict = self.loss_fn(model, data)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(1)
                    # set description
                    desc = f'Epoch: {epoch_counter} '
                    for k, v in loss_dict.items():
                        desc += f'{k}: {v.item():.4f} '
                    
                    pbar.set_description(desc) 
                    wandb.log(loss_dict)  
                    del loss_dict, loss
                    n_iter += 1
            
            # validate the model 
            valid_loss_dict = self._validate(model, val_dataset)
            print('Validation', epoch_counter)
            for k, v in valid_loss_dict.items():
                print(f'{k}: {v:.4f}')
            wandb.log(valid_loss_dict)

            if valid_loss_dict['Validation energy rmse'] < best_valid_loss:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))
                gc.collect() # free memory

            valid_n_iter += 1
        
        del train_loader, train_dataset, val_dataset
        gc.collect()
        
        start_time = time.time()
        self._test(model, test_dataset, ckpt_dir)
        print('test duration:', time.time() - start_time)
        wandb.finish()

    def _load_weights(self, model):
        try:
            state_dict = torch.load(os.path.join(self.config['load_model'], 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            if self.config['reset_head']:
                model.reset_head()
            print("Loaded pre-trained model with success from path: ",self.config['load_model'] )
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, val_dataset):
        pred_es, pred_fs = [], []
        label_es, label_fs = [], []
        model.eval()
        val_loader = PyGDataLoader(
                val_dataset, batch_size=64, num_workers=4,
                shuffle=False, drop_last=False, 
                worker_init_fn=lambda k: self._seed_everything(self.seed + (k * 1000)),
            )

        for bn, data in enumerate(val_loader):
            pred_e, pred_f,_ , __ = self.loss_fn(model, data)
            # If not water
            pred_e = pred_e * self.e_std + self.e_mean
            pred_f = pred_f * self.e_std


            e = data.e
            f = data.f

            pred_es.extend(pred_e.flatten().detach().cpu().numpy())
            label_es.extend(e.flatten().cpu().numpy())
            pred_fs.extend(pred_f.flatten().detach().cpu().numpy())
            label_fs.extend(f.flatten().cpu().numpy())
            del pred_e, pred_f, e, f
            
        energy_rmse = mean_squared_error(label_es, pred_es, squared=False)
        force_rmse = mean_squared_error(label_fs, pred_fs, squared=False)
        del pred_es, pred_fs, label_es, label_fs
        torch.cuda.empty_cache()
        gc.collect() # free memory
        # del val_loader
        model.train()
        return {
            'Validation energy rmse': energy_rmse,
            'Validation force rmse': force_rmse,

        }
    
    def _test(self, model, test_dataset, ckpt_path = None):
        model_path = os.path.join(self.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))
        test_loader = PyGDataLoader(
            test_dataset, batch_size=256, num_workers=self.config['training']['num_workers'],
            shuffle=False, drop_last=False,
            worker_init_fn=lambda k: self._seed_everything(self.seed + (k * 1000)),
            )

        pred_es, pred_fs = [], []
        label_es, label_fs = [], []
        model.eval()

        for bn, data in enumerate(test_loader):                
            pred_e, pred_f,_ , __ = self.loss_fn(model, data)
            # If not water
            pred_e = pred_e * self.e_std + self.e_mean
            pred_f = pred_f * self.e_std


            e = data.e
            f = data.f

            
            pred_es.extend(pred_e.flatten().detach().cpu().numpy())
            label_es.extend(e.flatten().cpu().numpy())
            pred_fs.extend(pred_f.flatten().detach().cpu().numpy())
            label_fs.extend(f.flatten().cpu().numpy())
            
            del pred_e, pred_f, e, f
                    
        
        energy_rmse = mean_squared_error(label_es, pred_es, squared=False)
        energy_mae = mean_absolute_error(label_es, pred_es)

        force_rmse = mean_squared_error(label_fs, pred_fs, squared=False)
        force_mae = mean_absolute_error(label_fs, pred_fs)

        # plot energy and force, also show their R-squared
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].scatter(label_es, pred_es, s=1)
        ax[0].plot([0, 1], [0, 1], transform=ax[0].transAxes, c='k', ls='--')
        ax[0].set_xlabel('True Energy')
        ax[0].set_ylabel('Predicted Energy')

        ax[1].scatter(label_fs, pred_fs, s=1)
        ax[1].plot([0, 1], [0, 1], transform=ax[1].transAxes, c='k', ls='--')
        ax[1].set_xlabel('True Force')
        ax[1].set_ylabel('Predicted Force')

        plt.tight_layout()
        # add it to wandb
        wandb.log({
            'Test energy rmse': energy_rmse,
            'Test energy mae': energy_mae,
            'Test force rmse': force_rmse,
            'Test force mae': force_mae,
            'Test plot': wandb.Image(fig),
        }
        )
        plt.close()

            

if __name__ == "__main__":
    config = yaml.load(open("config_water_fine_tune.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()