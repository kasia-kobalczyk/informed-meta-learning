import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = '/mnt/data/knk25.data/informed-meta-learning/data/'

class SetKnowledgeTrendingSinusoids(Dataset):
    def __init__(
            self, split='train', 
            root=f'{ROOT}/trending-sinusoids', 
            knowledge_type='full', 
            split_file='trending_sinusoids_splits'
        ):
        self.data = pd.read_csv(f'{root}/trending_sinusoids.csv')
        self.knowledge = pd.read_csv(f'{root}/trending_sinusoids_knowledge.csv')
        self.value_cols = [c for c in self.data.columns if c.isnumeric()]
        self.dim_x = 1
        self.dim_y = 1
        if split_file is None:
            split_file = 'trending_sinusoids_splits'
        self.train_test_val_split = pd.read_csv(f'{root}/{split_file}.csv')
        self.split = split
        self.knowledge_type = knowledge_type
        self.knowledge_input_dim = 4

        self._split_data()

    def _split_data(self):
        if self.split == 'train':
            train_ids = self.train_test_val_split[self.train_test_val_split.split == 'train'].curve_id
            self.data = self.data[self.data.curve_id.isin(train_ids)]
        elif self.split == 'val' or self.split == 'valid':
            val_ids = self.train_test_val_split[self.train_test_val_split.split == 'val'].curve_id
            self.data = self.data[self.data.curve_id.isin(val_ids)]
        elif self.split == 'test':
            test_ids = self.train_test_val_split[self.train_test_val_split.split == 'test'].curve_id
            self.data = self.data[self.data.curve_id.isin(test_ids)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        y = self.data.iloc[idx, :][self.value_cols].values
        x = np.linspace(-2, 2, len(y))
        curve_id = self.data.iloc[idx]['curve_id']
        
        knowledge = self.get_knowledge(curve_id)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) #[bs,  num_points, x_size]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) #[bs,  num_points, x_size]

        return x, y, knowledge


    def get_knowledge(self, curve_id):
        knowledge = self.knowledge[self.knowledge.curve_id == curve_id].drop('curve_id', axis=1).values
        knowledge = torch.tensor(knowledge, dtype=torch.float32).reshape(3, 1)
        indicator = torch.eye(3)
        knowledge = torch.cat([indicator, knowledge], dim=1)

        if self.knowledge_type == 'abc':
            revealed = np.random.choice([0, 1, 2])
            mask = torch.zeros((3, 1))
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == 'abc2':
            mask = torch.zeros((3, 1))
            num_revealed = np.random.choice([1, 2])
            revealed = np.random.choice([0, 1, 2], num_revealed, replace=False)
            mask[revealed] = 1.0
            knowledge = knowledge * mask
        elif self.knowledge_type == 'a':
            knowledge = knowledge[0, :].unsqueeze(0)
        elif self.knowledge_type == 'b':
            knowledge = knowledge[1, :].unsqueeze(0)
        elif self.knowledge_type == 'c':
            knowledge = knowledge[2, :].unsqueeze(0)
        elif self.knowledge_type == 'full':
            pass
        elif self.knowledge_type  == 'none':
            knowledge = torch.zeros_like(knowledge)
        else:
            raise NotImplementedError

        return knowledge
