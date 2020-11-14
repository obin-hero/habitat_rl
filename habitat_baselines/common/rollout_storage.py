#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
import numpy as np
import cv2

class RolloutStorage:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        OBS_LIST = []
    ):
        self.observations = {}
        self.OBS_LIST = OBS_LIST
        for sensor in observation_space.spaces:
            if sensor in OBS_LIST:
                self.observations[sensor] = torch.zeros(
                    num_steps + 1,
                    num_envs,
                    *observation_space.spaces[sensor].shape
                )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            if sensor in self.OBS_LIST:
                self.observations[sensor][self.step + 1].copy_(
                    observations[sensor]
                )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                if advantages is not None:
                    adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            if advantages is not None:
                adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
               recurrent_hidden_states_batch, 1
            )
            #recurrent_hidden_states_batch = self._flatten_helper(T,N,recurrent_hidden_states_batch)
            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            if advantages is not None : adv_targ = self._flatten_helper(T, N, adv_targ)
            else: adv_targ = None

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])

class RolloutStorage_HER:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        OBS_LIST = []
    ):
        self.observations = {}
        self.OBS_LIST = OBS_LIST
        self.num_envs = num_envs
        for sensor in observation_space.spaces:
            if sensor in OBS_LIST:
                self.observations[sensor] = torch.zeros(
                    num_steps + 1,
                    num_envs,
                    *observation_space.spaces[sensor].shape
                )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)



        self.re_observations = {}
        for sensor in observation_space.spaces:
            if sensor in OBS_LIST:
                self.re_observations[sensor] = torch.zeros(
                    (num_steps + 1) * num_envs,
                    *observation_space.spaces[sensor].shape
                )
        self.re_rewards = torch.zeros(num_steps * num_envs, 1)
        self.re_value_preds = torch.zeros((num_steps + 1)*num_envs, 1)
        self.re_returns = torch.zeros((num_steps + 1)*num_envs, 1)
        self.re_recurrent_hidden_states = torch.zeros(
            (num_steps + 1)*num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )
        self.re_action_log_probs = torch.zeros((num_steps) * num_envs, 1)
        self.re_actions = torch.zeros(num_steps * num_envs, action_shape)
        self.re_prev_actions = torch.zeros((num_steps + 1)*num_envs, action_shape)
        self.re_masks = torch.zeros((num_steps + 1)*num_envs, action_shape)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

        for sensor in self.re_observations:
            self.re_observations[sensor] = self.re_observations[sensor].to(device)
        self.re_rewards = self.re_rewards.to(device)
        self.re_value_preds = self.re_value_preds.to(device)
        self.re_returns = self.re_returns.to(device)
        self.re_action_log_probs = self.re_action_log_probs.to(device)
        self.re_actions = self.re_actions.to(device)
        self.re_prev_actions = self.re_prev_actions.to(device)
        self.re_masks = self.re_masks.to(device)
        self.re_recurrent_hidden_states = self.re_recurrent_hidden_states.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            if sensor in self.OBS_LIST:
                self.observations[sensor][self.step + 1].copy_(
                    observations[sensor]
                )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def rearrange_rollout(self):
        # B, step

        # for sensor in observations:
        #     if sensor in self.OBS_LIST:
        #         self.observations[sensor][self.step + 1].copy_(
        #             observations[sensor]
        #         )
        # self.recurrent_hidden_states[self.step + 1].copy_(
        #     recurrent_hidden_states
        # )
        # self.actions[self.step].copy_(actions)
        # self.prev_actions[self.step + 1].copy_(actions)
        # self.action_log_probs[self.step].copy_(action_log_probs)
        # self.value_preds[self.step].copy_(value_preds)
        # self.rewards[self.step].copy_(rewards)
        # self.masks[self.step + 1].copy_(masks)
        #
        # self.step = self.step + 1
        # collect episode indices
        done_step, env_idx = torch.where(self.masks.squeeze(-1) == 0)
        episode_in_envs = [[] for _ in range(self.num_envs)]
        for step, b in zip(done_step, env_idx):
            episode_in_envs[b].append([b, step])
        episodes = []
        for b, eps in enumerate(episode_in_envs):
            num_of_episodes = len(eps)
            new_episodes = []
            for episode_idx in range(num_of_episodes):
                episode_start = int(eps[episode_idx][1])
                episode_end = int(eps[episode_idx + 1][1]) if episode_idx != num_of_episodes - 1 else self.step
                if episode_start > episode_end:
                    print('h')
                new_episodes.append([b, episode_start, episode_end-1])
            if num_of_episodes > 0 :
                episodes.extend(new_episodes)

        # check each episodes
        for ep_id, ep in enumerate(episodes):
            b, start_idx, end_idx = ep
            start_x, _, start_y = self.observations['position'][start_idx, b]
            end_x, _, end_y = self.observations['position'][end_idx, b]
            dist = torch.sqrt((abs(end_x - start_x))**2 + (abs(end_y - start_y))**2)
            if dist < 2.0:
                dists = torch.norm(self.observations['position'][start_idx:end_idx, b] - self.observations['position'][start_idx, b],dim=1)
                if (dists > 2.0).any() and int(dists.argmax()) >= episodes[ep_id][1]:
                    episodes[ep_id][2] = int(dists.argmax())
                else:
                    episodes[ep_id] = None

        fake_step = 0
        for ep_id, ep in enumerate(episodes):
            if ep is not None:
                b, start_idx, end_idx = ep
                if start_idx > end_idx : continue
                target_rgb = self.observations['panoramic_rgb'][end_idx,b]
                target_depth = self.observations['panoramic_depth'][end_idx,b]
                target_goal = torch.cat((target_rgb/255., target_depth),2)
                target_pose = self.observations['position'][end_idx, b]
                dists = torch.norm(self.observations['position'][start_idx:end_idx+1, b] - target_pose,dim=1)
                if not (dists>1.0).any(): continue
                try:
                    last_idx = int(torch.where(dists > 1.0)[0].max()) + start_idx
                except:
                    print('ssssss')
                length = last_idx - start_idx + 1
                # for t in range(length):
                #     rgb = self.observations['panoramic_rgb'][start_idx+t,b].cpu().numpy().astype(np.uint8)
                #     end = (target_goal[:,:,:3]*255).cpu().numpy().astype(np.uint8)#self.observations['panoramic_rgb'][end_idx,b].cpu().numpy().astype(np.uint8)
                #     cv2.imshow('a', np.concatenate([rgb,end],0))
                #     cv2.waitKey(0)
                for sensor in self.observations:
                    if 'target_goal' in sensor:
                        self.re_observations[sensor][fake_step:fake_step+length] = target_goal
                    else:
                        self.re_observations[sensor][fake_step:fake_step+length] = self.observations[sensor][start_idx:last_idx+1, b]
                for t in range(length):
                    reward_t = max(dists[t] - dists[t+1], 0.0) * 0.2 - 0.01
                    self.re_rewards[fake_step+t] = reward_t
                    self.re_actions[fake_step+t] = self.actions[start_idx+t,b]
                    self.re_prev_actions[fake_step+t] = self.prev_actions[start_idx+t,b]
                    self.re_recurrent_hidden_states[fake_step+t] = self.recurrent_hidden_states[start_idx+t,:,b]
                    self.re_masks[fake_step+t] = 1.0
                self.re_rewards[fake_step+t] = 10.0
                self.re_masks[fake_step+t] = 0.0
                fake_step += length
        self.fake_step = fake_step
        # from habitat.utils.visualizations.utils import append_text_to_image
        # for t in range(fake_step):
        #     rgb = self.re_observations['panoramic_rgb'][t].cpu().int().numpy().astype(np.uint8)
        #     target_goal = (self.re_observations['target_goal'][t][:, :, :3] * 255).cpu().int().numpy().astype(np.uint8)
        #     view_img = np.concatenate([rgb, target_goal], 1)
        #     text = 't %d: act: %d reward %.3f mask: %d'%(t, int(self.re_actions[t]), self.re_rewards[t], self.re_masks[t])
        #     view_img = append_text_to_image(view_img, text)
        #     cv2.imshow('hi', view_img)
        #     cv2.waitKey(0)
        #TODO 1 Value Prediction
        #TODO 2 action log probs

    def compute_rearranged_returns(self, agent, gamma, tau):
        with torch.no_grad():
            hidden_states = self.re_recurrent_hidden_states[0].unsqueeze(1)
            for step in range(self.fake_step):
                last_observation = {
                    k: v[step].unsqueeze(0) for k, v in self.re_observations.items()
                }
                value, action_log_probs, _, hidden_states, *_= agent.evaluate_actions(
                    last_observation,
                    hidden_states,
                    self.re_prev_actions[step].unsqueeze(0),
                    self.re_masks[step].unsqueeze(0),
                    self.re_actions[step].unsqueeze(0)
                )
                self.re_value_preds[step] = value[0].detach()
                self.re_action_log_probs[step] = action_log_probs[0].detach()

        gae = 0
        for step in reversed(range(self.fake_step)):
            delta = (
                    self.re_rewards[step]
                    + gamma * self.re_value_preds[step + 1] * self.re_masks[step + 1]
                    - self.re_value_preds[step]
            )
            gae = delta + gamma * tau * self.re_masks[step + 1] * gae
            self.re_returns[step] = gae + self.re_value_preds[step]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                if advantages is not None:
                    adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            if advantages is not None:
                adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )
            recurrent_hidden_states_batch = self._flatten_helper(T,N,recurrent_hidden_states_batch)
            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            if advantages is not None : adv_targ = self._flatten_helper(T, N, adv_targ)
            else: adv_targ = None

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


    def recurrent_generator_her(self, advantages, num_mini_batch, advantages_her=None):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        stst = [[0,start_ind] for start_ind in range(0, num_processes, num_envs_per_batch)]
        batch_size = self.num_steps * num_envs_per_batch
        for i in range(int(np.ceil(self.fake_step/batch_size))):
            stst.append([1,i])
        for k in stst:
            mode, start_ind = k
            if mode == 0:
                observations_batch = defaultdict(list)

                recurrent_hidden_states_batch = []
                actions_batch = []
                prev_actions_batch = []
                value_preds_batch = []
                return_batch = []
                masks_batch = []
                old_action_log_probs_batch = []
                adv_targ = []

                for offset in range(num_envs_per_batch):
                    ind = perm[start_ind + offset]

                    for sensor in self.observations:
                        observations_batch[sensor].append(
                            self.observations[sensor][: self.step, ind]
                        )

                    recurrent_hidden_states_batch.append(
                        self.recurrent_hidden_states[0, :, ind]
                    )

                    actions_batch.append(self.actions[: self.step, ind])
                    prev_actions_batch.append(self.prev_actions[: self.step, ind])
                    value_preds_batch.append(self.value_preds[: self.step, ind])
                    return_batch.append(self.returns[: self.step, ind])
                    masks_batch.append(self.masks[: self.step, ind])
                    old_action_log_probs_batch.append(
                        self.action_log_probs[: self.step, ind]
                    )
                    if advantages is not None:
                        adv_targ.append(advantages[: self.step, ind])

                T, N = self.step, num_envs_per_batch

                # These are all tensors of size (T, N, -1)
                for sensor in observations_batch:
                    observations_batch[sensor] = torch.stack(
                        observations_batch[sensor], 1
                    )

                actions_batch = torch.stack(actions_batch, 1)
                prev_actions_batch = torch.stack(prev_actions_batch, 1)
                value_preds_batch = torch.stack(value_preds_batch, 1)
                return_batch = torch.stack(return_batch, 1)
                masks_batch = torch.stack(masks_batch, 1)
                old_action_log_probs_batch = torch.stack(
                    old_action_log_probs_batch, 1
                )
                if advantages is not None:
                    adv_targ = torch.stack(adv_targ, 1)

                # States is just a (num_recurrent_layers, N, -1) tensor
                recurrent_hidden_states_batch = torch.stack(
                    recurrent_hidden_states_batch, 1
                )
                #recurrent_hidden_states_batch = self._flatten_helper(T,N,recurrent_hidden_states_batch)
                # Flatten the (T, N, ...) tensors to (T * N, ...)
                for sensor in observations_batch:
                    observations_batch[sensor] = self._flatten_helper(
                        T, N, observations_batch[sensor]
                    )

                actions_batch = self._flatten_helper(T, N, actions_batch)
                prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
                value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
                return_batch = self._flatten_helper(T, N, return_batch)
                masks_batch = self._flatten_helper(T, N, masks_batch)
                old_action_log_probs_batch = self._flatten_helper(
                    T, N, old_action_log_probs_batch
                )
                if advantages is not None : adv_targ = self._flatten_helper(T, N, adv_targ)
                else: adv_targ = None
            else:
                observations_batch = defaultdict(list)
                end = min((start_ind+1)*batch_size,self.fake_step)
                for sensor in self.re_observations:
                    observations_batch[sensor] = self.re_observations[sensor][:end]

                recurrent_hidden_states_batch = self.re_recurrent_hidden_states[start_ind*batch_size].unsqueeze(1)
                actions_batch = self.re_actions[:end]
                prev_actions_batch = self.re_prev_actions[:end]
                value_preds_batch = self.re_value_preds[:end]
                return_batch = self.re_returns[:end]
                masks_batch = self.re_masks[:end]
                old_action_log_probs_batch = self.re_action_log_probs[:end]
                if advantages is not None:
                    adv_targ = advantages_her[:end]

                # T, N = self.step, num_envs_per_batch

                # Flatten the (T, N, ...) tensors to (T * N, ...)
                # for sensor in observations_batch:
                #     observations_batch[sensor] = self._flatten_helper(
                #         T, N, observations_batch[sensor]
                #     )
                #
                # actions_batch = self._flatten_helper(T, N, actions_batch)
                # prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
                # value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
                # return_batch = self._flatten_helper(T, N, return_batch)
                # masks_batch = self._flatten_helper(T, N, masks_batch)
                # old_action_log_probs_batch = self._flatten_helper(
                #     T, N, old_action_log_probs_batch
                # )
                # if advantages is not None:
                #     adv_targ = self._flatten_helper(T, N, adv_targ)
                # else:
                #     adv_targ = None

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )