import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.inp import INP
import torch
from config import Config

EVAL_CONFIGS = {
    'test_num_z_samples': 32,
    'knowledge_dropout': 0,
    'batch_size': 25,
    'device' : torch.device("cuda:{}".format(0))
}

def _load_model(config, save_dir, load_it='best'):
    print(save_dir)
    model = INP(config)
    model.to(config.device)
    model.eval()
    state_dict = torch.load(f'{save_dir}/model_{load_it}.pt')
    model.load_state_dict(state_dict)
    return model

def load_model(save_dir, load_it='best'):
    config = Config()
    config = config.from_toml(f'{save_dir}/config.toml')
    config.__dict__.update(EVAL_CONFIGS)
    model = _load_model(config, save_dir, load_it)
    return model, config


def get_mask(k_type):
    if k_type == 'a':
            mask = torch.tensor([1, 0, 0]).reshape(3, 1)
    elif k_type == 'b':
            mask =  torch.tensor([0, 1, 0]).reshape(3, 1)
    elif k_type == 'c' :
            mask = torch.tensor([0, 0, 1]).reshape(3, 1)
    elif k_type == 'abc':
            mask = torch.tensor([1, 1, 1]).reshape(3, 1)
    elif k_type == 'ab':
            mask = torch.tensor([1, 1, 0]).reshape(3, 1)
    elif k_type == 'ac':
            mask = torch.tensor([1, 0, 1]).reshape(3, 1)
    elif k_type == 'bc':
            mask = torch.tensor([0, 1, 1]).reshape(3, 1)

    elif k_type == 'a1':
            mask = torch.tensor([1, 0, 0, 0, 0]).reshape(5, 1)
    elif k_type == 'a2':
            mask = torch.tensor([0, 1, 0, 0, 0]).reshape(5, 1)
    elif k_type == 'b1':
            mask = torch.tensor([0, 0, 1, 0, 0]).reshape(5, 1)
    elif k_type == 'b2':
            mask = torch.tensor([0, 0, 0, 1, 0]).reshape(5, 1)
    elif k_type == 'w':
            mask = torch.tensor([0, 0, 0, 0, 1]).reshape(5, 1)

    elif k_type == 'raw':
            mask = None
        
    return mask


def plot_predictions(ax, i, outputs, x_context, y_context, x_target, extras, color='C0', plot_true=True):
    mean = outputs[0].mean[:, i].cpu()
    stddev = outputs[0].stddev[:, i].cpu()
    
    for j in range(min(mean.shape[0], 10)):
        ax.plot(x_target[i].flatten().cpu(), mean[j].flatten(), color=color, alpha=0.8)
        ax.fill_between(
            x_target[i].flatten().cpu(), 
            (mean[j] - stddev[j]).flatten(),
            (mean[j] + stddev[j]).flatten(),
            alpha=0.1,
            color=color
        )
    ax.scatter(x_context[i].flatten().cpu(), y_context[i].flatten().cpu(), color='black')
    if plot_true:
        ax.plot(extras['x'][i].flatten().cpu(), extras['y'][i].flatten().cpu(), color='black', linestyle='--', alpha=0.8)
