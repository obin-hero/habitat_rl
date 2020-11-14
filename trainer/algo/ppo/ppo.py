#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
EPS_PPO = 1e-5
import time
TIME_DEBUG = False
def log_time(prev_time, log):
    print("[TIME] ", log, time.time() - prev_time)
    return time.time()

class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def get_advantages_her(self,rollouts):
        advantages = rollouts.re_returns[:-1] - rollouts.re_value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)
    def update(self, rollouts, her=False):
        if TIME_DEBUG: s = time.time()

        advantages = self.get_advantages(rollouts)
        if her:
            advantages_her = self.get_advantages_her(rollouts)
        if TIME_DEBUG: s = log_time(s, 'get_advanta')
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        il_loss_epoch = 0
        aux_loss1_epoch = 0
        aux_loss2_epoch = 0

        for e in range(self.ppo_epoch):
            if her:
                data_generator = rollouts.recurrent_generator_her(
                    advantages, self.num_mini_batch, advantages_her
                )
            else:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            if TIME_DEBUG: s = log_time(s, 'recuurent generate')
            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample
                # print(
                #       'hid', recurrent_hidden_states_batch.shape,
                #       'act', actions_batch.shape,
                #       'prev', prev_actions_batch.shape,
                #       'value_pred', value_preds_batch.shape,
                #       'ret', return_batch.shape,
                #       'masks', masks_batch.shape,
                #       'old', old_action_log_probs_batch.shape,
                #       'adv_targ', adv_targ.shape,
                # )
                if TIME_DEBUG: s = log_time(s, 'get sample')
                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    act_distribution,
                    pred_aux1,
                    pred_aux2
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )
                if TIME_DEBUG: s = log_time(s, 'evaluate action')
                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()
                if TIME_DEBUG: s = log_time(s, 'loss')
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                if pred_aux1 is not None:
                    aux_loss1 = F.binary_cross_entropy_with_logits(pred_aux1, obs_batch['have_been'].float())
                if pred_aux2 is not None:
                    aux_loss2 = F.mse_loss(F.sigmoid(pred_aux2), obs_batch['target_dist_score'].float())

                if TIME_DEBUG: s = log_time(s, 'clip value los')

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )
                if pred_aux1 is not None:
                    total_loss += aux_loss1
                    aux_loss1_epoch += aux_loss1.item()
                if pred_aux2 is not None:
                    total_loss += aux_loss2
                    aux_loss2_epoch += aux_loss2.item()

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()
                if TIME_DEBUG: s = log_time(s, 'backward and step')
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                if TIME_DEBUG: s = log_time(s, 'sum loss, entropy')

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        il_loss_epoch /= num_updates
        aux_loss1_epoch /= num_updates
        aux_loss2_epoch /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, il_loss_epoch, aux_loss1_epoch, aux_loss2_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass
