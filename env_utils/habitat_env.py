#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Habitat environment without Dataset
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
import os
from habitat_sim.utils.common import quat_to_coeffs
import quaternion as q
import time

MAX_DIST = 3.0
MIN_DIST = 1.0

class Env:
    r"""Fundamental environment class for `habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside `Env`. Acts as a base for other derived environment
    classes. `Env` consists of three major components: ``dataset`` (`episodes`), ``simulator`` (`sim`) and `task` and connects all the three components
    together.
    """

    observation_space: SpaceDict
    action_space: SpaceDict
    _config: Config
    _dataset: Optional[Dataset]
    _episodes: List[Type[Episode]]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Type[Episode]]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze()."
        )
        self._config = config
        self._current_episode_index = None
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }

        self._scenes = config.DATASET.CONTENT_SCENES
        self._swap_building_every = config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES
        print('[HabitatEnv] Total {} scenes : '.format(len(self._scenes)), self._scenes)
        print('[HabitatEnv] swap building every',self._swap_building_every)
        self._current_scene_episode_idx = 0
        self._current_scene_idx = 0

        self._config.defrost()
        if 'mp3d' in config.DATASET.DATA_PATH:
            self._config.SIMULATOR.SCENE = os.path.join(config.DATASET.SCENES_DIR, 'mp3d/{}/{}.glb'.format(self._scenes[0],self._scenes[0]))
        else:
            self._config.SIMULATOR.SCENE = os.path.join(config.DATASET.SCENES_DIR,
                                                        'gibson_habitat/{}.glb'.format(self._scenes[0]))
            if not os.path.exists(self._config.SIMULATOR.SCENE):
                self._config.SIMULATOR.SCENE = os.path.join(self._config.DATASET.SCENES_DIR,
                                                            'gibson_more/{}.glb'.format(self._scenes[0]))       
        self._config.freeze()

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,

            config=self._config.TASK,
            sim=self._sim
        )
        self.observation_space = SpaceDict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

        #TODO listup demonstration data
        self._episode_dataset = {}
        # scene별로 정리해두는게 좋아요
        #self._episode_dataset = {'scene1': [scene1_episode1.dat.gz,scene1_episode2.dat.gz,scene1_episode3.dat.gz....],
        #                     'scene2': [scene2_episode1.dat.gz,scene2_episode2.dat.gz,scene2_episode3.dat.gz....],
        #                     ....}


    @property
    def current_episode(self) -> Type[Episode]:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Type[Episode]) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return None

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def get_next_episode(self, episode_id, scene_id):
        scene_name = scene_id.split('/')[-1][:-4]
        if self._current_scene_episode_idx >= len(self._episode_datasets[scene_name]):
            self._current_scene_episode_idx = 0
        episode = self._episode_datasets[scene_name][self._current_scene_episode_idx]
        goals = []
        goal = NavigationGoal(**{'position': episode['goal_position']})
        goals.append(goal)
        self.start_position = episode['start_position']
        self.start_rotation = episode['start_rotation']

        episode_info = {'episode_id': self._current_scene_episode_idx,
                      'scene_id': scene_id,
                      'start_position': self.start_position,
                      'start_rotation': self.start_rotation.components,
                      'goals': goals,
                      'start_room': None,
                      'shortest_paths': None}
        episode = NavigationEpisode(**episode_info)
        #TODO load demonstration data
        self.demonstration = []
        #print(episode)
        return episode
    def reset(self) -> Observations:
        """Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()
        scene_name = self._scenes[self._current_scene_idx]

        if self._current_scene_iter >= self._swap_building_every:
            self._episode_iterator[scene_name] = self._current_scene_episode_idx + 1
            self._current_scene_idx = (self._current_scene_idx + 1)%len(self._scenes)
            scene_name = self._scenes[self._current_scene_idx]
            if scene_name not in self._episode_iterator.keys():
                self._episode_iterator.update({scene_name:0})
            self._current_scene_episode_idx = self._episode_iterator[scene_name]
            self._current_scene_iter = 0
            self._config.defrost()
            if 'mp3d' in self._config.DATASET.DATA_PATH:
                self._config.SIMULATOR.SCENE = os.path.join(self._config.DATASET.SCENES_DIR, 'mp3d/{}/{}.glb'.format(scene_name,scene_name))
            else:
                self._config.SIMULATOR.SCENE = os.path.join(self._config.DATASET.SCENES_DIR,
                                                        'gibson_habitat/{}.glb'.format(scene_name))
            self._config.freeze()
            self.reconfigure(self._config)
            print('swapping building %s, every episode will be sampled in : %f, %f'%(scene_name, self.MIN_DIST, self.MAX_DIST))


        self._current_episode = self.get_next_episode(self._current_scene_episode_idx, self._config.SIMULATOR.SCENE)
        self._config.defrost()

        self._config.SIMULATOR.AGENT_0.START_POSITION = self._current_episode.start_position
        if isinstance(self._current_episode.start_rotation, np.ndarray):
            self._config.SIMULATOR.AGENT_0.START_ROTATION = quat_to_coeffs(q.from_float_array(self._current_episode.start_rotation)).tolist()
        else:
            self._config.SIMULATOR.AGENT_0.START_ROTATION = quat_to_coeffs(self._current_episode.start_rotation).tolist()
        self._config.SIMULATOR.AGENT_0.IS_SET_START_STATE = True
        self._config.freeze()
        self.reconfigure(self._config)

        self._current_scene_episode_idx += 1
        self._current_scene_iter += 1
        observations = self.task.reset(episode=self._current_episode)
        self._task.measurements.reset_measures(
            episode=self._current_episode, task=self.task
        )
        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to `action_space`) to be performed
            inside the environment. Action is a name or index of allowed
            task's action and action arguments (belonging to action's
            `action_space`) to support parametrized and continuous actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        # Support simpler interface as well
        if isinstance(action, str) or isinstance(action, (int, np.integer)):
            action = {"action": action}

        observations = self.task.step(
            action=action, episode=self.current_episode
        )
        self._task.measurements.update_measures(
            episode=self.current_episode, action=action, task=self.task
        )
        self._update_step_stats()

        #TODO include demonstration in observations
        observations.update(self.demonstration)

        return observations

    def seed(self, seed: int) -> None:
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config
        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over `Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: `get_reward_range()`, `get_reward()`, `get_done()`, `get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct `Env`
        :param dataset: dataset to construct `Env`.
        """

        self._env = Env(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Type[Episode]]:
        return self._env.episodes

    @property
    def current_episode(self) -> Type[Episode]:
        return self._env.current_episode

    @episodes.setter
    def episodes(self, episodes: List[Type[Episode]]) -> None:
        self._env.episodes = episodes

    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the `step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()
