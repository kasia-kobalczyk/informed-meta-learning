#%%
%load_ext autoreload
%autoreload 2
from utils import load_model, get_mask
import sys
import os
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from datasets.datasets import *
from datasets.utils import get_dataloader
from models.loss import NLL
from evaluation.utils import plot_predictions

import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

plt.style.use('science')
sns.set_palette('Dark2')
#%%
# Load the models
save_dirs = {
    'base' : '../saves/trend_sinusoid_base_0',
    'informed' : '../saves/trend_sinusoid_informed_0',
}

models = list(save_dirs.keys())
model_dict = {}
config_dict = {}

for model_name, save_dir in save_dirs.items():
    model_dict[model_name], config_dict[model_name] = load_model(save_dir, load_it='best')
    model_dict[model_name].eval()

model_names = list(model_dict.keys())
# %%
# Setup the dataloaders
config = Namespace(
      min_num_context=0,
      max_num_context=100,
      num_targets=100,
      noise=0.2,
      batch_size=25,
      x_sampler='uniform',
      test_num_z_samples=32,
      dataset='set-trending-sinusoids',
      device='cuda:0'
  )

dataset = SetKnowledgeTrendingSinusoids(
  root='../data/trending-sinusoids', split='test', knowledge_type='full'
)
data_loader = get_dataloader(dataset, config)

# %%
# Evaluate the models on different knowledge types
loss = NLL()

losses = {}
outputs_dict = {}
data_knowledge = {}
eval_type_ls = ['raw', 'informed',  'a', 'b', 'c', 'ab', 'bc', 'ac']

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
        
summary_df['print_value'] = summary_df['mean'].apply(lambda x: f'{x:.1f}') #+ ' \scriptsize{(' + summary_df['std'].apply(lambda x: f'{x:.1f}') + ')}'
print_df = summary_df.dropna(subset=['mean']).pivot(
    columns='num_context', index=['model_name', 'eval_type'], values=['print_value']
).T.round(2)

print_df.droplevel(0, axis=0).dropna(axis=1, how='all')
# %%
for batch in data_loader:
    (x_context, y_context), (x_target, y_target), full_knowledge, extras = batch
    x_context = x_context.to(config.device)
    y_context = y_context.to(config.device)
    x_target = x_target.to(config.device)
    y_target = y_target.to(config.device)

sample_idx = np.random.choice(list(range(x_target.shape[-2])), max(num_context_ls))

num_context_ls = [0, 1, 3, 5]

fig, axs = plt.subplots(len(num_context_ls), 4, figsize=(10, 6))

for j, knowledge_type in enumerate(['raw', 'a', 'b', 'c']):
    if knowledge_type == 'raw':
        knowledge = None
    else:
        mask = get_mask(knowledge_type)
        knowledge = full_knowledge * mask
    
    for i, num_context in enumerate(num_context_ls):
        x_context = x_target[:, sample_idx[:num_context], :]
        y_context = y_target[:, sample_idx[:num_context], :]

        with torch.no_grad():
            informed_outupts = model_dict['informed'](
                x_context, y_context, x_target, y_target=y_target, knowledge=knowledge)

        plot_predictions(
            axs[i][j], 0, informed_outupts, x_context, y_context, x_target, extras, color=f'C{j}', plot_true=True
        )
# %%
