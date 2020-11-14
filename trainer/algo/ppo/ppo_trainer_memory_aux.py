#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
#from habitat.utils.visualizations.utils import observations_to_image
from utils.vis_utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
#from habitat_baselines.common.env_utils import construct_envs
from env_utils.make_env_utils import construct_envs
from env_utils import *
#GraphMemoryEnv = VisTargetGraphMemEnv
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from trainer.algo.ppo import PPO
from model.resnet.resnet_policy import PointNavResNetPolicy, ExploreResNetPolicy
from model.policy import *
import time
TIME_DEBUG = False
ADD_IL = False
def log_time(prev_time, log):
    print("[TIME] ", log, time.time() - prev_time)
    return time.time()
from trainer.algo.ppo.ppo_trainer_memory import PPOTrainer_Memory
import torch.nn.functional as F
@baseline_registry.register_trainer(name="custom_ppo_memory_aux")
class PPOTrainer_Memory_aux(PPOTrainer_Memory):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]
    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                _,
                preds,
                _
            ) = self.actor_critic.act(
                self.last_observations,
                self.last_recurrent_hidden_states,
                self.last_prev_actions,
                self.last_masks,
            )
        actions = actions.unsqueeze(1)
        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        pred1, pred2 = preds
        if pred1 is not None:
            have_been = F.sigmoid(pred1[:,0]).detach().cpu().numpy().tolist()
        else:
            have_been = None
        if pred2 is not None:
            pred_target_distance = F.sigmoid(pred2[:,0]).detach().cpu().numpy().tolist()
        else:
            pred_target_distance = None
        log_strs = []
        for i in range(len(actions)):
            hb = have_been[i] if have_been is not None else -1
            ptd = pred_target_distance[i] if pred_target_distance is not None else -1
            log_str = 'have_been: %.3f pred_dist: %.3f'%(hb, ptd)
            log_strs.append(log_str)
        self.envs.call(['log_info']*len(have_been),[{'log_type':'str', 'info':log_strs[i]} for i in range(len(have_been))])
        #scenes = [curr_ep.scene_id.split('/')[-2] for curr_ep in self.envs.current_episodes()]xdd/d
        if self.collect_mode == 'RL':
            k = [a[0] for a in actions.cpu().numpy()]
            batch, rewards, dones, infos = self.envs.step(k)
            #self.envs.render('human')
        else:
            k = self.last_observations['gt_action'].cpu().long().numpy().tolist()
            batch, rewards, dones, infos = self.il_envs.step(k)
            #self.il_envs.render('human')
        env_time += time.time() - t_step_env


        t_update_stats = time.time()
        #batch = batch_obs(observations, device=self.device)

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        if self.collect_mode == 'RL':
            current_episode_reward += rewards
            running_episode_stats["reward"] += (1 - masks) * current_episode_reward
            running_episode_stats["count"] += 1 - masks
            for k, v in self._extract_scalars_from_infos(infos).items():
                try:
                    v = torch.tensor(
                        v, dtype=torch.float, device=current_episode_reward.device
                    ).unsqueeze(1)
                    if k not in running_episode_stats:
                        running_episode_stats[k] = torch.zeros_like(
                            running_episode_stats["count"]
                        )
                    running_episode_stats[k] += (1 - masks) * v
                except:
                    print('EEEEERRROR!!!!', masks.shape, v.shape)
                    print('key:', k)
               
            current_episode_reward *= masks
        if self._static_encoder:
            with torch.no_grad():
                pass
                #batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            {k: v[:self.num_processes] for k,v in batch.items()},
            recurrent_hidden_states[:,:self.num_processes],
            actions[:self.num_processes],
            actions_log_probs[:self.num_processes],
            values[:self.num_processes],
            rewards[:self.num_processes],
            masks[:self.num_processes],
        )
        self.last_observations = batch
        self.last_recurrent_hidden_states = recurrent_hidden_states#.to(self.device)
        self.last_prev_actions = actions
        self.last_masks = masks.to(self.device)
        pth_time += time.time() - t_update_stats
        return pth_time, env_time, self.num_processes
