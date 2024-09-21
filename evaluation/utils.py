import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.inp import INP
import torch
from config import Config
from models.loss import NLL
import numpy as np
import pandas as pd

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


def get_summary_df(model_dict, config_dict, data_loader, eval_type_ls, model_names):
    # Evaluate the models on different knowledge types
    loss = NLL()

    losses = {}
    outputs_dict = {}
    data_knowledge = {}

    num_context_ls = [0, 1, 3, 5, 10, 15]
    for model_name in model_names:
        losses[model_name] = {}
        outputs_dict[model_name] = {}
        for eval_type in eval_type_ls:
            losses[model_name][eval_type] = {}
            outputs_dict[model_name][eval_type] = {}
            for num_context in num_context_ls:
                losses[model_name][eval_type][num_context] = []
                outputs_dict[model_name][eval_type][num_context] = []
        
    knowledge_ls = []    
    y_target_ls = []
    extras_ls = []

    for model_name in model_names:

        model, config =  model_dict[model_name], config_dict[model_name]

        for batch in data_loader:
            (x_context, y_context), (x_target, y_target), knowledge, extras = batch
            knowledge_ls.append(knowledge)
            y_target_ls.append(y_target)
            extras_ls.append(extras)
            x_context = x_context.to(config.device)
            y_context = y_context.to(config.device)
            x_target = x_target.to(config.device)
            y_target = y_target.to(config.device)
            
            sample_idx = np.random.choice(list(range(x_target.shape[-2])), max(num_context_ls))

            for _ in range(3):
                for num_context in num_context_ls:
                    x_context = x_target[:, sample_idx[:num_context], :]
                    y_context = y_target[:, sample_idx[:num_context], :]
                
                    for eval_type in eval_type_ls:
                        with torch.no_grad():      
                            if eval_type == 'raw':
                                outputs = model(
                                    x_context,
                                    y_context,
                                    x_target,
                                    y_target=y_target,
                                    knowledge=None
                                )
                            elif config.use_knowledge:
                                if eval_type == 'informed':
                                    outputs = model(
                                    x_context,
                                    y_context,
                                    x_target,
                                    y_target=y_target,
                                    knowledge=knowledge
                                    )
                                else:
                                    mask = get_mask(eval_type)
                                    outputs = model(
                                    x_context,
                                    y_context,
                                    x_target,
                                    y_target=y_target,
                                    knowledge=knowledge * mask
                                    )
                            else:
                                continue
                            outputs = tuple([o.cpu() if isinstance(o, torch.Tensor) else o for o in outputs])
                            loss_value = loss.get_loss(outputs[0], outputs[1], outputs[2], outputs[3], y_target)
                            losses[model_name][eval_type][num_context].append(loss_value)
                            outputs_dict[model_name][eval_type][num_context].append(outputs)
            

    loss_summary = {}
    for model_name in model_names:
        loss_summary[model_name] = {}
        for eval_type in eval_type_ls:
            loss_summary[model_name][eval_type] = {}
            for num_context in num_context_ls:
                loss_summary[model_name][eval_type][num_context] = {}
                loss_values = losses[model_name][eval_type][num_context]
                if len(loss_values) == 0:
                    loss_summary[model_name][eval_type][num_context]['mean'] = np.nan
                    loss_summary[model_name][eval_type][num_context]['std'] = np.nan
                else:
                    loss_values = torch.concat(loss_values, dim=0)
                    loss_summary[model_name][eval_type][num_context]['median'] = torch.median(loss_values).item()
                    loss_summary[model_name][eval_type][num_context]['mean'] = torch.mean(loss_values).item()
                    loss_summary[model_name][eval_type][num_context]['std'] = torch.std(loss_values).item()

    summary_df = pd.DataFrame()
    for model_name in model_names:
        for eval_type in eval_type_ls:
            df = pd.DataFrame().from_dict(loss_summary[model_name][eval_type], orient='index')
            df['num_context'] = df.index
            df['model_name'] = model_name 
            df['eval_type'] = eval_type
            summary_df = pd.concat([summary_df, df])


    return summary_df, losses

    