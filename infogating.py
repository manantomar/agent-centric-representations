# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ig_utils import InfoNCE, UNet, LinearPolicy, RandomShiftsAug, Encoder

import random
import utils
import os

import matplotlib.pyplot as plt

def shuffle_batch(x):
    '''
    shuffle tensor x along the batch dim (ie 0).
    '''
    idx = torch.randperm(x.size(0))
    x_shuf = x[idx]
    return x_shuf

class InfoGatingAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 augmentation=RandomShiftsAug(pad=4), lam=0.1):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.lam = lam # IG sparsity parameter

        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        
        self.mask_net = UNet(
            num_blocks=int(np.log2(84)-1),
            img_size=84,
            filter_start=32,
            in_chnls=9,
            out_chnls=3,
            norm='gn').to(device)             

        # Linear actor that learns to act over the latent rep. using bc loss
        self.policy = LinearPolicy(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        # Inv Dynamics model, uses a contrastive form
        self.inv_model = InfoNCE(2*feature_dim+action_shape[0], action_shape[0], 1).to(device)
        
        self.encoder_opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.inv_model.parameters()), lr=lr)
        self.mask_opt = torch.optim.Adam(self.mask_net.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # data augmentation
        self.aug = augmentation

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.mask_net.train(training)
        self.inv_model.train(training)
        self.policy.train(training)

    def get_latent(self, obs, mask, step=0):
        mask_reshape = (1 * torch.as_tensor(mask.reshape(3, 84, 84), device=self.device))
        mask_reshape = torch.repeat_interleave(mask_reshape, 3, dim=0)
        nz = 255 * torch.zeros_like(mask_reshape)
        mask_obs = (obs * mask_reshape) + (nz * (1. - mask_reshape))

        # eval on unmasked observations
        return self.encoder(obs.unsqueeze(0))

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)

        mask = F.relu(torch.tanh(self.mask_net(obs.unsqueeze(0))))
        latent = self.get_latent(obs, mask, step)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.policy(latent.float(), stddev)
    
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_buffer)
        obs, action, reward, discount, next_obs, k_step, obs_k = utils.to_torch(
            batch, self.device)

        # augment
        obs_aug = self.aug(obs.float())
        obs = self.aug(obs.float())
        obs_k = self.aug(obs_k.float())

        mask = F.relu(torch.tanh(self.mask_net(obs, step)))
        mask_k = F.relu(torch.tanh(self.mask_net(obs_k, step)))

        mask_reshape = (1. * mask.reshape(obs.shape[0], 3, 84, 84))
        mask_reshape_aug = (1. * mask_k.reshape(obs_k.shape[0], 3, 84, 84))

        mask_reshape = torch.repeat_interleave(mask_reshape, 3, dim=1)
        mask_reshape_aug = torch.repeat_interleave(mask_reshape_aug, 3, dim=1)

        nz = 255 * torch.zeros_like(mask_reshape)
        mask_obs = (obs * mask_reshape) + (nz * (1. - mask_reshape))
        mask_obs_k = (obs_k * mask_reshape_aug) + (nz * (1. - mask_reshape_aug))
        
        noise_loss = 0.5 * mask.float().view(mask.shape[0], -1).mean() + 0.5 * mask_k.float().view(mask.shape[0], -1).mean()
        noise_var_loss = 0.5 * (mask.float().view(mask.shape[0], -1)**2.).mean() + 0.5 * (mask_k.float().view(mask_k.shape[0], -1)**2.).mean()
        summed_mask = torch.stack([mask_reshape[i].sum([-1, -2]) / 84**2 for i in range(3)], dim=1)

        mask_obs_org = mask_obs.clone()

        # Randomly mixing masked and unmasked inputs
        k = int(obs.size(0) * 0.5)
        perm = torch.randperm(obs.size(0))
        idx = perm[:k]
        mask_obs[idx] = obs[idx]

        perm = torch.randperm(obs_k.size(0))
        idx = perm[:k]
        mask_obs_k[idx] = obs_k[idx]

        ## Warm start where noise is removed
        if step < 5000:
            inv_loss = self.inv_model(self.encoder(mask_obs.detach()), self.encoder(mask_obs_k.detach()), action)
            ig_loss = inv_loss - 0.1 * noise_loss
        ## Info-Gate learning where most info is removed, i.e. noise is added maximally
        else:
            inv_loss = self.inv_model(self.encoder(mask_obs), self.encoder(mask_obs_k), action)
            ig_loss = inv_loss + self.lam * noise_loss + 1e-3 * noise_var_loss #+ 0.5 * self.entropy(summed_mask)

        # optimize mask_net and encoder
        self.mask_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        ig_loss.backward()
        self.encoder_opt.step()
        self.mask_opt.step()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.policy(self.encoder(mask_obs).detach(), stddev)
        pred_action = dist.sample(clip=self.stddev_clip)

        # offline BC Loss
        actor_loss = F.mse_loss(pred_action, action)

        # optimize actor and encoder
        self.policy_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.policy_opt.step()

        if step % 5000 == 0:
            self.plot_obs(mask_obs_org[:5], step) # plot masked obs

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        return metrics

    def entropy(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        return b.sum()

    def plot_obs(self, obs, step):
        def show(img):
            npimg = img.numpy().astype(dtype='uint8')
            fig = plt.imshow(np.transpose(npimg, (1,2,0))[:, :, :], interpolation='nearest')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            return fig
        
        # Plot all frames individually
        fig = show(obs[0, :3, :, :].cpu().data +0.5, ).figure
        fig.savefig(os.path.join(os.getcwd(), 'obs1_{}.png'.format(step)), format='png')

        fig = show(obs[0, 3:6, :, :].cpu().data +0.5, ).figure
        fig.savefig(os.path.join(os.getcwd(), 'obs2_{}.png'.format(step)), format='png')

        fig = show(obs[0, 6:9, :, :].cpu().data +0.5, ).figure
        fig.savefig(os.path.join(os.getcwd(), 'obs3_{}.png'.format(step)), format='png')
