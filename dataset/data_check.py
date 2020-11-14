import os
import cv2
import joblib
import matplotlib.pyplot as plt
DATA_DIR = '/media/obin/5d368da0-d601-490b-b5d8-6122946470b8/DATA/vistarget_demo2/'
train_data_list = [os.path.join(DATA_DIR+'train/medium',x) for x in os.listdir(DATA_DIR+'train/medium')]
val_data_list = [os.path.join(DATA_DIR+'val/medium',x) for x in os.listdir(DATA_DIR+'val/medium')]
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from gym.spaces.dict_space import Dict as SpaceDict

from habitat.config import Config
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "9"
from habitat_sim.utils.common import quat_to_coeffs
import quaternion as q
import habitat_sim
import habitat
from gym.spaces.dict_space import Dict as SpaceDict
from gym.spaces.box import Box
from tqdm import tqdm
import pickle

SPLIT = 'train'
if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
import joblib

MAX_DIST = np.Inf
MIN_DIST = 1.5

NEAR_DIST_TH = 1.5
MIDDLE_DIST_TH = 3.0

MEDIUM_MAX_FAR_DIST = 5.0

import numpy as np


def add_panoramic_camera(task_config):
    task_config.SIMULATOR.RGB_SENSOR_LEFT = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_LEFT.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_LEFT.ORIENTATION = [0, 2 / 3 * np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_LEFT.ANGLE = "left"
    task_config.SIMULATOR.RGB_SENSOR_RIGHT = task_config.SIMULATOR.RGB_SENSOR.clone()
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.TYPE = "PanoramicPartRGBSensor"
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.ORIENTATION = [0, -2 / 3 * np.pi, 0]
    task_config.SIMULATOR.RGB_SENSOR_RIGHT.ANGLE = "right"
    task_config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'RGB_SENSOR_LEFT', 'RGB_SENSOR_RIGHT']

    task_config.SIMULATOR.DEPTH_SENSOR_LEFT = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.SIMULATOR.DEPTH_SENSOR_LEFT.TYPE = "PanoramicPartDepthSensor"
    task_config.SIMULATOR.DEPTH_SENSOR_LEFT.ORIENTATION = [0, 2 / 3 * np.pi, 0]
    task_config.SIMULATOR.DEPTH_SENSOR_LEFT.ANGLE = "left"
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT.TYPE = "PanoramicPartDepthSensor"
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT.ORIENTATION = [0, -2 / 3 * np.pi, 0]
    task_config.SIMULATOR.DEPTH_SENSOR_RIGHT.ANGLE = "right"
    task_config.SIMULATOR.AGENT_0.SENSORS += ['DEPTH_SENSOR', 'DEPTH_SENSOR_LEFT', 'DEPTH_SENSOR_RIGHT']

    task_config.TASK.CUSTOM_VISTARGET_SENSOR = habitat.Config()
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.TYPE = 'CustomVisTargetSensor'

    task_config.TASK.PANORAMIC_SENSOR = habitat.Config()
    task_config.TASK.PANORAMIC_SENSOR.TYPE = 'PanoramicRGBSensor'
    task_config.TASK.PANORAMIC_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.WIDTH
    task_config.TASK.PANORAMIC_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT
    task_config.TASK.PANORAMIC_DEPTH_SENSOR = task_config.SIMULATOR.DEPTH_SENSOR.clone()
    task_config.TASK.PANORAMIC_DEPTH_SENSOR.TYPE = 'PanoramicDepthSensor'
    task_config.TASK.PANORAMIC_DEPTH_SENSOR.WIDTH = task_config.SIMULATOR.DEPTH_SENSOR.WIDTH
    task_config.TASK.PANORAMIC_DEPTH_SENSOR.HEIGHT = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT

    if "STOP" not in task_config.TASK.POSSIBLE_ACTIONS:
        task_config.TASK.SUCCESS.TYPE = "Success_woSTOP"
    task_config.TASK.SUCCESS.SUCCESS_DISTANCE = task_config.TASK.SUCCESS_DISTANCE

    return task_config


class DataCollectEnv:
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
        self._scenes = config.DATASET.CONTENT_SCENES
        self._swap_building_every = config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES
        self._current_scene_episode_idx = 0
        self._current_scene_idx = 0

        self._config.defrost()
        self._config.SIMULATOR.SCENE = os.path.join(config.DATASET.SCENES_DIR,
                                                    'mp3d/{}/{}.glb'.format(self._scenes, self._scenes))
        self._config.freeze()

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )

    def validate_data(self, data):
        data_len = len(data['rgb'])
        changed = False
        for i in range(data_len):
            position = data['position'][i]
            rotation = q.from_float_array(data['rotation'][i])
            obs = self._sim.get_observations_at(position, rotation)
            new_rgb, new_depth = self.process_obs(obs)
            old_rgb, old_depth = data['rgb'][i], data['depth'][i]
            if not (new_rgb[:,1:-1] == old_rgb).all():
                #past_view, future_view = data['rgb'][max(0,i-1)], data['rgb'][min(data_len-1, i+1)]
                #compare = np.concatenate([past_view, future_view],0)[:,:,[2,1,0]]
                #curr_view = np.concatenate([old_rgb, new_rgb[:,1:-1]], 0)[:,:,[2,1,0]]
                #cv2.imshow('hi', np.concatenate([curr_view, compare],1))
                #cv2.waitKey(0)
                data['rgb'][i] = new_rgb[:,1:-1]
                data['depth'][i] = new_depth[:,1:-1]
                changed = True
        return changed, data

    def process_obs(self, obs):
        rgb = np.concatenate([obs['rgb_left'], obs['rgb'], obs['rgb_right']], 1)
        depth = np.concatenate([obs['depth_left'], obs['depth'], obs['depth_right']], 1)
        return rgb, depth


def collect_data(config):  # 1 env per 1 config
    # np.random.seed(config.SEED)
    scene_name = config.DATASET.CONTENT_SCENES
    env = DataCollectEnv(config)
    split = config.DATASET.SPLIT
    if split == 'train':
        scene_data_list = [x for x in train_data_list if scene_name in x]
    elif split == 'val':
        scene_data_list = [x for x in val_data_list if scene_name in x]
    for data_file in tqdm(scene_data_list):
        data = joblib.load(data_file)
        changed, new_data = env.validate_data(data)
        if changed:
            joblib.dump(new_data,data_file)

    env._sim.close()

    return

splits = ['val']

from IL_configs.default import get_config
import numpy as np
from multiprocessing import Pool
import cv2
from env_utils.vistarget_nav_task import CustomVisTargetSensor

for split in splits:
    config = get_config('IL_configs/base.yaml')
    configs = []
    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.defrost()
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.freeze()

    dataset = make_dataset('PointNav-v1')
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    # 17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r 29hnd4uzFmX 2n8kARJN3HM 5LpN3gDmAk7 5q7pvUzZiYa 759xd9YjKW5 7y3sRwLe3Va 82sE5b5pLXE
    # 8WUmhLawc2A B6ByNegPMKs D7G3Y4RVNrH D7N2EKCX4Sj E9uDoFAP3SH EDJbREhghzL GdvgFV5R1Z5 HxpKQynjfin JF19kD82Mey JeFG25nYj2p
    # JmbYfDe2QKZ
    valid_scenes = []
    for scene_name in scenes:
        if split == 'train':
            scene_data_list = [x for x in train_data_list if scene_name in x]
        elif split == 'val':
            scene_data_list = [x for x in val_data_list if scene_name in x]
        changed = False
        for each_data in scene_data_list:
            valid = ((time.time() - os.stat(each_data).st_mtime) / 3600) > 24
            print(scene_name, (time.time() - os.stat(each_data).st_mtime) / 3600)
            if not valid:
                changed = True
                break
        if not changed:
            valid_scenes.append(scene_name)
    scenes = valid_scenes
    for i in range(len(scenes)):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.DATASET.CONTENT_SCENES = scenes[i]

        task_config = add_panoramic_camera(task_config)

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        proc_config.freeze()
        configs.append(proc_config.TASK_CONFIG)


    # process map IL_configs
    num_thread = 7
    start = time.time()
    with Pool(num_thread) as p:
        p.map(collect_data, configs, int(len(configs) / num_thread))
    end = time.time() - start