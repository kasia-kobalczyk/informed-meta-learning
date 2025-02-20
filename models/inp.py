import torch.nn as nn
import torch
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.modules import XYEncoder, LatentEncoder, Decoder, XEncoder
from models.utils import MultivariateNormalDiag


class INP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.xy_encoder = XYEncoder(config)
        self.latent_encoder = LatentEncoder(config)
        self.decoder = Decoder(config)
        self.x_encoder = XEncoder(config)
        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples

    def forward(self, x_context, y_context, x_target, y_target, knowledge=None):
        x_context = self.x_encoder(x_context)  # [bs, num_context, x_transf_dim]
        x_target = self.x_encoder(x_target)  # [bs, num_context, x_transf_dim]

        R = self.encode_globally(x_context, y_context, x_target)

        z_samples, q_z_Cc, q_zCct = self.sample_latent(
            R, x_context, x_target, y_target, knowledge
        )
        # reshape z_samples to the shape of x_target
        R_target = self.target_dependent_representation(R, x_target, z_samples)

        p_yCc = self.decode_target(x_target, R_target)

        return p_yCc, z_samples, q_z_Cc, q_zCct

    def encode_globally(self, x_context, y_context, x_target):
        """
        Encode context set all together
        """
        R = self.xy_encoder(x_context, y_context, x_target)

        if x_context.shape[1] == 0:
            R = torch.zeros((R.shape[0], 1, R.shape[-1])).to(R.device)

        return R

    def get_knowledge_embedding(self, knowledge):
        return self.latent_encoder.get_knowledge_embedding(knowledge)

    def sample_latent(self, R, x_context, x_target, y_target, knowledge):
        """
        Sample latent variable z given the global representation
        (and during training given the target)
        """
        q_zCc = self.infer_latent_dist(R, knowledge, x_context.shape[1])

        if y_target is not None and self.training:
            R_from_target = self.encode_globally(x_target, y_target, x_target)
            q_zCct = self.infer_latent_dist(R_from_target, knowledge, x_target.shape[1])
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        if self.training:
            z_samples = sampling_dist.rsample([self.train_num_z_samples])
        else:
            z_samples = sampling_dist.rsample([self.test_num_z_samples])
        # z_samples.shape = [n_z_samples, bs, 1, z_dim]
        return z_samples, q_zCc, q_zCct

    def infer_latent_dist(self, R, knowledge, n):
        """
        Infer the latent distribution given the global representation
        """
        q_z_stats = self.latent_encoder(R, knowledge, n)
        q_z_loc, q_z_scale = q_z_stats.split(self.config.hidden_dim, dim=-1)
        q_z_scale = 0.01 + 0.99 * F.softplus(q_z_scale)
        q_zCc = MultivariateNormalDiag(q_z_loc, q_z_scale)
        return q_zCc

    def target_dependent_representation(self, R, x_target, z_samples):
        """
        Compute the target dependent representation of the context set
        """
        R_target = z_samples  # [num_z_samples, batch_size, 1, hidden_dim]

        # [num_z_samples, batch_size, num_targets, hidden_dim]

        R_target = R_target.expand(-1, -1, x_target.shape[1], -1)

        return R_target

    def decode_target(self, x_target, R_target):
        """
        Decode the target set given the target dependent representation
        """
        p_y_stats = self.decoder(x_target, R_target)

        p_y_loc, p_y_scale = p_y_stats.split(self.config.output_dim, dim=-1)

        # bound the variance (minimum 0.1)
        p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)

        p_yCc = MultivariateNormalDiag(p_y_loc, p_y_scale)

        return p_yCc


if __name__ == "__main__":
    from argparse import Namespace
    from loss import ELBOLoss
    from dataset.utils import get_dataloader
    from dataset.datasets import SetKnowledgeTrendingSinusoids
    import numpy as np
    import random

    config = Namespace(
        # model
        input_dim=1,
        output_dim=1,
        xy_encoder_num_hidden=2,
        xy_encoder_hidden_dim=128,
        data_agg_func="mean",
        latent_encoder_num_hidden=2,
        decoder_hidden_dim=64,
        decoder_num_hidden=2,
        decoder_activation="gelu",
        hidden_dim=128,
        x_transf_dim=128,
        x_encoder_num_hidden=1,
        test_num_z_samples=32,
        train_num_z_samples=1,
        knowledge_extractor_num_hidden=0,
        knowledge_dropout=0,
        knowledge_dim=128,
        knowledge_merge="sum",
        text_encoder="set",
        use_knowledge=True,
        freeze_llm=True,
        tune_llm_layer_norms=False,
        # dataset
        batch_size=64,
        min_num_context=1,
        max_num_context=30,
        x_sampler="uniform",
        noise=0,
        # reproducibility
        seed=44,
        dataset="set-trending-sinusoids",
        num_targets=50,
    )
    config.device = "cpu"

    dataset = SetKnowledgeTrendingSinusoids(split="train", knowledge_type="abc2")
    train_dataloader = get_dataloader(dataset, config)
    config.knowledge_input_dim = dataset.knowledge_input_dim

    model = INP(config)
    loss_func = ELBOLoss()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    for i, batch in enumerate(train_dataloader):
        print(i)
        context, target, knowledge, _ = batch
        x_context, y_context = context
        x_target, y_target = target

        if config.use_knowledge:
            outputs = model(x_context, y_context, x_target, y_target, knowledge)
        else:
            outputs = model(x_context, y_context, x_target, y_target, None)

        print(y_target.shape)
        p_yCc = outputs[0]
        print(p_yCc.mean.shape)

        loss = loss_func(outputs, y_target)

        print(loss)
