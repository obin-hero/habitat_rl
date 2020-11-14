#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, RGBSensor, DepthSensor, SemanticSensor
from habitat.core.dataset import Dataset, Episode
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
import quaternion as q
import time
import torch
# def make_panoramic(left, front, right, torch_tensor=False):
#     if not torch_tensor: return np.concatenate([left, front, right],1)[:,1:-1]
#     else: return torch.cat((left, front, right),1)[:,1:-1]
import habitat_sim
@registry.register_sensor(name="PanoramicPartRGBSensor")
class PanoramicPartRGBSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.COLOR
        super().__init__(config=config)


    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "rgb_" + self.angle

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )
    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs[:,:,:3]




@registry.register_sensor(name="PanoramicPartSemanticSensor")
class PanoramicPartSemanticSensor(RGBSensor):
    def __init__(self, config, **kwargs: Any):
        self.config = config
        self.angle = config.ANGLE
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC
        super().__init__(config=config)


    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "semantic_" + self.angle

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.uint8,
        )
    def get_observation(self, obs, *args: Any, **kwargs: Any) -> Any:
        obs = obs.get(self.uuid, None)
        return obs
@registry.register_sensor(name="PanoramicPartDepthSensor")
class PanoramicPartDepthSensor(DepthSensor):
    def __init__(self, config):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH
        self.angle = config.ANGLE

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "depth_" + self.angle

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    # This is called whenver reset is called or an action is taken
    def get_observation(self, obs,*args: Any, **kwargs: Any):
        obs = obs.get(self.uuid, None)
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = np.expand_dims(
                obs, axis=2
            )  # make depth observation a 3D array
        else:
            obs = obs.clamp(self.config.MIN_DEPTH, self.config.MAX_DEPTH)

            obs = obs.unsqueeze(-1)

        if self.config.NORMALIZE_DEPTH:
            # normalize depth observation to [0, 1]
            obs = (obs - self.config.MIN_DEPTH) / (
                self.config.MAX_DEPTH - self.config.MIN_DEPTH
            )

        return obs


@registry.register_sensor(name="PanoramicRGBSensor")
class PanoramicRGBSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim = sim
        super().__init__(config=config)
        self.config = config
        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA

        # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_rgb"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR
    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, 252, 3),
            dtype=np.uint8,
        )
    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        rgb_list = [observations['rgb_%d'%(i)] for i in range(self.num_camera)]
        rgb_array = np.concatenate(rgb_list,1)
        if rgb_array.shape[1] > self.config.HEIGHT*4:
            left = rgb_array.shape[1] - self.config.HEIGHT*4
            slice = left//2
            rgb_array = rgb_array[:,slice:slice+self.config.HEIGHT*4]
        return rgb_array
        #return make_panoramic(observations['rgb_left'],observations['rgb'],observations['rgb_right'], self.torch)

@registry.register_sensor(name="PanoramicDepthSensor")
class PanoramicDepthSensor(DepthSensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.DEPTH

        if config.NORMALIZE_DEPTH:
            self.min_depth_value = 0
            self.max_depth_value = 1
        else:
            self.min_depth_value = config.MIN_DEPTH
            self.max_depth_value = config.MAX_DEPTH

        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=self.min_depth_value,
            high=self.max_depth_value,
            shape=(self.config.HEIGHT, 252, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_depth"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.DEPTH

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['depth_%d'%(i)] for i in range(self.num_camera)]
        rgb_array = np.concatenate(depth_list,1)
        if rgb_array.shape[1] > self.config.HEIGHT*4:
            left = rgb_array.shape[1] - self.config.HEIGHT*4
            slice = left//2
            rgb_array = rgb_array[:,slice:slice+self.config.HEIGHT*4]
        return rgb_array
        #return make_panoramic(observations['depth_left'],observations['depth'],observations['depth_right'], self.torch)

@registry.register_sensor(name="PanoramicSemanticSensor")
class PanoramicSemanticSensor(SemanticSensor):
    def __init__(self, sim, config, **kwargs: Any):
        self.sim_sensor_type = habitat_sim.SensorType.SEMANTIC

        self.torch = False#sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA

        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=np.Inf,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32)

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "panoramic_semantic"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    # This is called whenver reset is called or an action is taken
    def get_observation(self, observations,*args: Any, **kwargs: Any):
        depth_list = [observations['semantic_%d'%(i)] for i in range(self.num_camera)]
        return np.concatenate(depth_list,1)
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.simulator import (
    AgentState,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import Success
@registry.register_measure(name='Success_woSTOP')
class Success_woSTOP(Success):
    r"""Whether or not the agent succeeded at its task

    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        if (
           distance_to_target < self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1.0
        else:
            self._metric = 0.0


@registry.register_measure(name='GoalIndex')
class GoalIndex(Measure):
    cls_uuid: str = "goal_index"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self.num_goals = len(episode.goals)
        self.goal_index = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        self._metric = {'curr_goal_index': self.goal_index,
                        'num_goals': self.num_goals}

    def increase_goal_index(self):
        self.goal_index += 1
        self._metric = {'curr_goal_index': self.goal_index,
                        'num_goals': self.num_goals}
        if self.goal_index >= self.num_goals:
            self._metric = {'curr_goal_index': self.goal_index-1,
                            'num_goals': self.num_goals}
            return True
        else: return False


@registry.register_measure(name='Custom_DistanceToGoal')
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        self.goal_idx = -1
        self.update_metric(episode=episode, *args, **kwargs)


    def update_metric(self, episode: Episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position

        if self.goal_idx == -1 : self.goal_idx = 0
        else:
            self.goal_idx = task.measurements.measures[GoalIndex.cls_uuid].get_metric()['curr_goal_index']

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.DISTANCE_TO == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    episode.goals[self.goal_idx].position,
                    episode,
                )
            elif self._config.DISTANCE_TO == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
                )

            self._previous_position = current_position
            self._metric = distance_to_target

@registry.register_measure(name='Custom_SPL')
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._episode_view_points = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        #rint(self._start_end_episode_distance)
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        #rint(self._agent_episode_distance, self._start_end_episode_distance)
        self._previous_position = current_position

        self._metric =(
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure(name='Custom_SoftSPL')
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to SPL with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "softspl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

@registry.register_sensor(name="CustomVisTargetSensor")
class CustomVisTargetSensor(Sensor):


    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        print(config)
        use_depth = True#'DEPTH_SENSOR_0' in self._sim.config.sim_cfg.AGENT_0.SENSORS
        use_rgb = True#'RGB_SENSOR_0' in self._sim.config.AGENT_0.SENSORS
        self.channel = use_depth + 3 * use_rgb
        self.height = config.HEIGHT
        self.width = 252
        self.curr_episode_id = -1
        self.curr_scene_id = ''
        self.torch = False#self._sim.config.HABITAT_SIM_V0.GPU_GPU
        self.num_camera = config.NUM_CAMERA
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "target_goal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=1.0, shape=(self.height, self.width, self.channel), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ) -> Optional[int]:

        episode_id = episode.episode_id
        scene_id = episode.scene_id
        if (self.curr_episode_id != episode_id) or (self.curr_scene_id != scene_id):
            self.curr_episode_id = episode_id
            self.curr_scene_id = scene_id
            self.goal_obs = []
            self.goal_pose = []
            for goal in episode.goals:
                position = goal.position
                euler = [0, 2 * np.pi * np.random.rand(), 0]
                rotation = q.from_rotation_vector(euler)
                obs = self._sim.get_observations_at(position,rotation)
                rgb_list = [obs['rgb_%d'%(i)] for i in range(self.num_camera)]
                rgb_array = np.concatenate(rgb_list, 1)/255.
                if rgb_array.shape[1] > self.height*4:
                    left = rgb_array.shape[1] - self.height*4
                    slice = left // 2
                    rgb_array = rgb_array[:, slice:slice + self.height*4]
                depth_list = [obs['depth_%d'%(i)] for i in range(self.num_camera)]
                depth_array = np.concatenate(depth_list, 1)
                if depth_array.shape[1] > self.height*4:
                    left = depth_array.shape[1] - self.height*4
                    slice = left // 2
                    depth_array = depth_array[:, slice:slice + self.height*4]
                if 'semantic_0' in obs.keys():
                    semantic_list = [obs['semantic_%d' % (i)] for i in range(self.num_camera)]
                    semantic_array = np.expand_dims(np.concatenate(semantic_list, 1),2)
                #rgb_array = make_panoramic(obs['rgb_left'], obs['rgb'], obs['rgb_right'], self.torch)/255.
                #depth_array = make_panoramic(obs['depth_left'], obs['depth'], obs['depth_right'], self.torch)
                if self.torch:
                    goal_obs = torch.cat([rgb_array, depth_array],2)
                else:
                    goal_obs = np.concatenate([rgb_array, depth_array],2)
                if 'semantic_0' in obs.keys():
                    goal_obs = np.concatenate([goal_obs, semantic_array],2)
                self.goal_obs.append(goal_obs)
                self.goal_pose.append([position, euler])
        return self.goal_obs


'''
from enum import Enum
class HighLevelActions(Enum):
    STOP = 0
    MOVE = 1
import attr
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
import habitat
@habitat.registry.register_action_space_configuration(name='HighLevelAction')
class HighLevelAction(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()
        config[HabitatSimActions.MOVE] = habitat_sim.ActionSpec("noisy_strafe_left",NoisyStrafeActuationSpec(0.25, noise_amount=0.05),)


@registry.register_task_action(name='HighLevelAction')
class HihghLevelAction(SimulatorTaskAction):
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "HIGH_LEVEL"

    def step(self, action):

        # 1 / 2 / 3 / 4 / 5 / 6
        dist = 1.0 #m

        angle = [0, 1/3*np.pi, 2/3*np.pi, np.pi, 4/3*np.pi, 5/3*np.pi, 2*np.pi]

        angle = [1/3*np.pi, 0, -1/3*np.pi, -2/3]

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return self._sim.get_observations_at()

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )

'''
