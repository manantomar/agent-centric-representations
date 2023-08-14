import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist

from utils import TruncatedNormal, weight_init, schedule

class RandomShiftsAug(nn.Module):
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3

        action_dim = 6
        self.repr_dim = 256 * 5 * 5
        self.repr_shape = (256, 5, 5)
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 128, 6, stride=6, padding=0),  # 14x14
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, 1, stride=1, padding=0),  # 14x14
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, 3, stride=1, padding=0),  # 12x12
                                     nn.ReLU(),
                                     nn.Conv2d(128, 256, 4, stride=2, padding=0),  # 5x5
                                     nn.ReLU())

        self.linear = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.linear(h)
        return h
    
class LinearPolicy(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_shape[0]))

        self.apply(weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist

def simple_nce_loss(z1, z2, temp=0.1, reduction=True):
    '''
    Compute basic infonce loss using embeddings z1/z2
    '''
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    attn_lsmax = F.log_softmax(torch.mm(z1, z2.T) / temp, dim=1)
    if reduction:
        nce_loss = -(torch.eye(z1.size(0), device=z1.device) * attn_lsmax).sum(dim=1).mean()
    else:
        nce_loss = -(torch.eye(z1.size(0), device=z1.device) * attn_lsmax).sum(dim=1)
    return nce_loss

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=50): #256):
        super().__init__()
        # hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class InfoNCE(nn.Module):
    def __init__(self, feature_dim, action_dim, num_actions=1):
        super().__init__()
        
        self.train_samples = 256
        self.action_dim = action_dim

        self.projector = projection_MLP(feature_dim, 256, 1)

        self.apply(weight_init)

    def forward(self, x1, x2, action, return_logits=False):
        self.device = x1.device
        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.sample(x1.size(0), action.size(1))
        
        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        fused = torch.cat([x1.unsqueeze(1).expand(-1, targets.size(1), -1), x2.unsqueeze(1).expand(-1, targets.size(1), -1), targets], dim=-1)
        B, N, D = fused.size()
        fused = fused.reshape(B * N, D)
        out = self.projector(fused)
        energy = out.view(B, N)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth.detach())

        if return_logits:
            return logits

        return loss

    def _sample(self, num_samples: int, action_size: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, action_size)
        samples = np.random.uniform(-1, 1, size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int, action_size: int) -> torch.Tensor:
        samples = self._sample(batch_size * self.train_samples, action_size)
        return samples.reshape(batch_size, self.train_samples, -1)

"""
Adapted from https://github.com/applied-ai-lab/genesis.
"""
import argparse
import torch.nn as nn

def flatten(x):
    return x.view(x.size(0), -1)

def unflatten(x):
    return x.view(x.size(0), -1, 1, 1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1, 1, 1)

class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        )

class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU(inplace=True)
        )

class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class UNet(nn.Module):

    def __init__(self, num_blocks, img_size=64,
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        super(UNet, self).__init__()
        # TODO(martin): make more general
        c = filter_start
        if norm == 'in':
            conv_block = ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]
        self.down = []
        self.up = []
        # 3x3 kernels, stride 1, padding 1
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2**(num_blocks-1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(2*c*self.featuremap_size**2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*c*self.featuremap_size**2), nn.ReLU()
        )
        if out_chnls > 0:
            self.final_conv = nn.Conv2d(c, out_chnls, 1)
        else:
            self.final_conv = nn.Identity()

        self.out_chnls = out_chnls

    def forward(self, x, step=5000):
        batch_size = x.size(0)
        x = x / 255.0 - 0.5
        x_down = [x]
        skip = []
        # Down
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
            x_down.append(act)
        # FC
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)
        outputs = [10, 21, 42, 84]
        idx = 0
        # Up
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, size=outputs[idx], mode='nearest')
            idx += 1
        mu = self.final_conv(x_up)

        if step > 1000:
            return mu

        std = 0.5
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)

        stddev = schedule('linear(0.1,0.1,25000)', 0)
        action = dist.sample(clip=0.3)
        return action

    def forward_ig(self, x, step=5000, only_enc=False):
        batch_size = x.size(0)
        x = x / 255.0 - 0.5
        x_down = [x]
        skip = []
        # Down
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest', recompute_scale_factor=True)
            x_down.append(act)
        # FC
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)
        if only_enc:
            return x_up
        outputs = [10, 21, 42, 84]
        idx = 0
        # Up
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, size=outputs[idx], mode='nearest')
            idx += 1
        mu = self.final_conv(x_up)

        if step > 1000:
            return mu

        std = 0.5
        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)

        stddev = schedule('linear(0.1, 0.1, 25000)', 0)
        action = dist.sample(clip=0.3)
        return action