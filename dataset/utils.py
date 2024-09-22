import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from dataset.dataset import *

def collate_fn(batch, kwargs, collate_knowledge=True):
    num_context_ls = list(range(kwargs['min_num_context'], kwargs['max_num_context']))
    num_context = np.random.choice(num_context_ls)

    x, y, knowledge = zip(*batch)
    num_samples = x[0].shape[0]
    x_size = x[0].shape[1]
    y_size = y[0].shape[1]
    batch_size = len(x)
    x = torch.stack(x)
    y = torch.stack(y)
    x_context = torch.zeros(batch_size,  num_context, x_size)
    y_context = torch.zeros(batch_size, num_context, y_size)
    x_target = torch.zeros(batch_size,  num_samples, x_size)
    y_target = torch.zeros(batch_size, num_samples, y_size)
    context_idx = torch.zeros(batch_size, num_context, 1).long()

    for i, (x_i, y_i) in enumerate(zip(x, y)):
        # random subsample
        if kwargs['x_sampler'] == 'uniform':
            sample_context = np.random.choice(x_i.shape[0], num_context, replace=False)
    
        elif kwargs['x_sampler'] == 'half-normal':
            sample_context = np.random.normal(loc=25, scale=20, size=num_context)
            sample_context = np.clip(sample_context, 0, x_i.shape[0] - 1).astype(int)

        elif kwargs['x_sampler'] == 'half-uniform':
            sample_context = np.random.uniform(low=0, high=x_i.shape[0] // 2, size=num_context)
            sample_context = sample_context.astype(int)
        
        elif kwargs['x_sampler'] == 'quarter-uniform':
            sample_context = np.random.uniform(low=0, high=x_i.shape[0] // 4, size=num_context)
            sample_context = sample_context.astype(int)

        elif kwargs['x_sampler'] == 'random-uniform':
            min = x_i.shape[0] // 4
            max = x_i.shape[0] - 1
            end = np.random.uniform(low=min, high=max, size=1).astype(int)
            sample_context = np.random.uniform(low=0, high=end, size=num_context)
            
        elif kwargs['x_sampler'].startswith('random-uniform-'):
            n = int(kwargs['x_sampler'].split('-')[-1])
            samples = np.random.uniform(low=0, high=x_i.shape[0] - 1, size=n)
            sample_context = sorted(samples)[:num_context]


        x_target[i, :, :] = x_i
        
        if kwargs['noise'] > 0 :
            # add observation noise
            y_target[i, :, :] = y_i + torch.normal(mean=0, std=kwargs['noise'], size=y_i.shape)

        else:
            y_target[i, :, :] = y_i

        x_context[i, :, :] = x_target[i, sample_context, :]
        y_context[i, :, :] = y_target[i, sample_context, :]

        context_idx[i, :, :] = torch.tensor(sample_context).unsqueeze(-1)

    
    extras = {
        'x': x,
        'y': y,
        'context_idx': context_idx
    }

    if collate_knowledge:
        knowledge = torch.stack(knowledge)

    return (x_context, y_context), (x_target, y_target), knowledge, extras


def get_dataloader(dataset, config):
    if config.dataset in ['set-trending-sinusoids', 'set-trending-sinusoids-dist-shift']:
        collate_knowledge = True
    else:
        collate_knowledge = False
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, config.__dict__, collate_knowledge)
    )
    return data_loader

def setup_dataloaders(config):
    extras = {}
    
    split_file = config.get('split_file')

    if config.dataset == 'set-trending-sinusoids':
        train_dataset = SetKnowledgeTrendingSinusoids(split='train', knowledge_type=config.knowledge_type)
        val_dataset = SetKnowledgeTrendingSinusoids(split='val', knowledge_type=config.knowledge_type)
        test_dataset = SetKnowledgeTrendingSinusoids(split='test', knowledge_type=config.knowledge_type)
       
    elif config.dataset == 'set-trending-sinusoids-dist-shift':
        train_dataset = SetKnowledgeTrendingSinusoidsDistShift(split='train', knowledge_type=config.knowledge_type)
        val_dataset = SetKnowledgeTrendingSinusoidsDistShift(split='val', knowledge_type=config.knowledge_type)
        test_dataset = SetKnowledgeTrendingSinusoidsDistShift(split='test', knowledge_type=config.knowledge_type)
    
    elif config.dataset == 'temperature':
        train_dataset = Temperatures(split='train', knowledge_type=config.knowledge_type)
        val_dataset = Temperatures(split='val', knowledge_type=config.knowledge_type)
        test_dataset = Temperatures(split='test', knowledge_type=config.knowledge_type)
    
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")

    extras['knowledge_input_dim'] = train_dataset.knowledge_input_dim

    
    assert(config.input_dim == test_dataset.dim_x)
    assert(config.output_dim == test_dataset.dim_y)

    train_dataloader = get_dataloader(train_dataset, config)
    val_dataloader = get_dataloader(val_dataset, config)
    test_dataloader = get_dataloader(test_dataset, config)

    return train_dataloader, val_dataloader, test_dataloader, extras


if __name__ == "__main__":
    import os
    import sys
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from dataset.dataset import *
    from argparse import Namespace

    config = Namespace(
        min_num_context=1,
        max_num_context=15,
        num_targets=50,
        noise=0,
        x_sampler='uniform',
        batch_size=4,
        dataset='temperatures',
        knowledge_type='set'
    )
    dataset = Temperatures(split='test', knowledge_type='llama_embed')
    data_loader = get_dataloader(dataset, config)

    for batch in data_loader:
        (x_context, y_context), (x_target, y_target), knowledge, extras = batch
        print(x_context.shape, y_context.shape, x_target.shape, y_target.shape)
        print(knowledge[0].shape)
        break
