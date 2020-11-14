#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from utils.visdommonitor import VisdomMonitor
from gym.wrappers.monitor import Wrapper

import os

import cv2
class EvalEnvWrapper(Wrapper):
    def __init__(self,env,directory='.', uid='0'):
        from habitat_baselines.common.utils import generate_video
        super().__init__(env)
        self.generate_video = generate_video
        self.video_dir = directory
        self.uid = uid
        self.number_of_episodes = 1000
    def setup_embedding_network(self, visual_encoder, prev_action_embedding):
        self.env.setup_embedding_network(visual_encoder, prev_action_embedding)
    @property
    def current_episode(self):
        return self.env.current_episode
    @property
    def episode_over(self):
        return self.env.episode_over
    def reset(self):
        obs = super().reset()
        self.img_frames = [self.render(mode='rgb_array')]
        return obs

    def setup_embedding_network(self, visual_encoder, prev_action_embedding):
        self.env.setup_embedding_network(visual_encoder, prev_action_embedding)
    def update_graph(self, node_list, affinity, changed_info, curr_info):
        self.env.update_graph(node_list, affinity, changed_info, curr_info)
    def draw_activated_nodes(self, activated_node_list):
        self.env.draw_activated_nodes(activated_node_list)
    def build_path_follower(self):
        self.env.build_path_follower()
    def get_best_action(self,goal=None):
        return self.env.get_best_action(goal)


    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.img_frames.append(self.render(mode='rgb_array'))
        '''        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.'''
        if done:
            if 'success' in info.keys():
                ep_info = {'episode': info['episode'], 'success': info['success'], 'spl': info['spl'],
                           'distance_to_goal': info['distance_to_goal'], 'length': info['length'],
                           'collisions': info['collisions']['count'],}
            if 'coverage' in info.keys():
                ep_info = {'episode': info['episode'], 'total_reward': info['total_reward'], 'coverage': info['coverage'], 'length': info['length'], 'collisions': info['collisions']['count'],}
            img_shape = (self.img_frames[0].shape[0],self.img_frames[0].shape[1])
            if img_shape[0]%16 != 0 or img_shape[1]%16 != 0 :
                required_img_shape = (16 * (img_shape[1]//16), 16 * (img_shape[0]//16))
            else:
                required_img_shape = (img_shape[1], img_shape[0])
            resized_img_frames = [cv2.resize(img, required_img_shape) for img in self.img_frames]
            self.img_frames = resized_img_frames

            self.generate_video(video_option=['disk'],
                               video_dir=self.video_dir,
                               images=self.img_frames,
                               episode_id=self.env.current_episode.episode_id,
                               checkpoint_idx='',
                               metrics= ep_info,
                               fps=30,
                                )
        return obs, reward, done, info

import numpy as np
def add_panoramic_camera(task_config, remain_front_rgbd=False, normalize_depth=True):
    num_of_camera = 360//task_config.SIMULATOR.RGB_SENSOR.HFOV
    assert isinstance(num_of_camera, int)
    angles = [2 * np.pi * idx/ num_of_camera for idx in range(num_of_camera-1,-1,-1)]
    half = num_of_camera//2
    angles = angles[half:] + angles[:half]
    sensors = []
    use_semantic = 'PANORAMIC_SEMANTIC_SENSOR' in task_config.TASK.SENSORS
    use_depth = 'PANORAMIC_DEPTH_SENSOR' in task_config.TASK.SENSORS
    for camera_idx in range(num_of_camera):
        curr_angle = angles[camera_idx]
        if curr_angle > 3.14:
            curr_angle -= 2 * np.pi
        new_camera_config = task_config.SIMULATOR.RGB_SENSOR.clone()
        new_camera_config.TYPE = "PanoramicPartRGBSensor"

        new_camera_config.ORIENTATION = [0, curr_angle, 0]
        new_camera_config.ANGLE = "{}".format(camera_idx)
        task_config.SIMULATOR.update({'RGB_SENSOR_{}'.format(camera_idx): new_camera_config})
        sensors.append('RGB_SENSOR_{}'.format(camera_idx))

        if use_depth:
            new_depth_camera_config = task_config.SIMULATOR.DEPTH_SENSOR.clone()
            new_depth_camera_config.TYPE = "PanoramicPartDepthSensor"
            new_depth_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_depth_camera_config.ANGLE = "{}".format(camera_idx)
            new_depth_camera_config.NORMALIZE_DEPTH = normalize_depth
            task_config.SIMULATOR.update({'DEPTH_SENSOR_{}'.format(camera_idx): new_depth_camera_config})
            sensors.append('DEPTH_SENSOR_{}'.format(camera_idx))
        if use_semantic:
            new_semantic_camera_config = task_config.SIMULATOR.SEMANTIC_SENSOR.clone()
            new_semantic_camera_config.TYPE = "PanoramicPartSemanticSensor"
            new_semantic_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_semantic_camera_config.ANGLE = "{}".format(camera_idx)
            task_config.SIMULATOR.update({'SEMANTIC_SENSOR_{}'.format(camera_idx): new_semantic_camera_config})
            sensors.append('SEMANTIC_SENSOR_{}'.format(camera_idx))

    if remain_front_rgbd:
        task_config.SIMULATOR.RGB_SENSOR.WIDTH = 256
        task_config.SIMULATOR.RGB_SENSOR.HEIGHT = 256
        task_config.SIMULATOR.RGB_SENSOR.HFOV = 90
        task_config.SIMULATOR.RGB_SENSOR.POSITION = [0,1.25,0]

        task_config.SIMULATOR.DEPTH_SENSOR.WIDTH = 256
        task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256
        task_config.SIMULATOR.DEPTH_SENSOR.HFOV = 90
        task_config.SIMULATOR.DEPTH_SENSOR.POSITION = [0, 1.25, 0]
        task_config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
        task_config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
        task_config.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
        sensors.extend(['RGB_SENSOR', 'DEPTH_SENSOR'])
    task_config.SIMULATOR.AGENT_0.SENSORS = sensors

    task_config.TASK.PANORAMIC_SENSOR = habitat.Config()
    task_config.TASK.PANORAMIC_SENSOR.TYPE = 'PanoramicRGBSensor'
    task_config.TASK.PANORAMIC_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.HEIGHT * 4
    task_config.TASK.PANORAMIC_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT
    task_config.TASK.PANORAMIC_SENSOR.NUM_CAMERA = num_of_camera
    if use_depth:
        task_config.TASK.PANORAMIC_DEPTH_SENSOR = task_config.SIMULATOR.DEPTH_SENSOR.clone()
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.TYPE = 'PanoramicDepthSensor'
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.WIDTH = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT * 4
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.HEIGHT = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.NUM_CAMERA = num_of_camera
    if use_semantic:
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR = habitat.Config()
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.TYPE = 'PanoramicSemanticSensor'
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.WIDTH = task_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT * 4
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.HEIGHT = task_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.NUM_CAMERA = num_of_camera

    task_config.TASK.CUSTOM_VISTARGET_SENSOR = habitat.Config()
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.TYPE = 'CustomVisTargetSensor'
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.NUM_CAMERA = num_of_camera
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.HEIGHT * 4
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT
   
    if "STOP" not in task_config.TASK.POSSIBLE_ACTIONS:
        task_config.TASK.SUCCESS.TYPE = "Success_woSTOP"
    task_config.TASK.SUCCESS.SUCCESS_DISTANCE = task_config.TASK.SUCCESS_DISTANCE
    return task_config

def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]], rank: int, kwargs
) -> Union[Env, RLEnv]:

    print('make-env')
    env = env_class(config=config)
    env.seed(rank)
    if kwargs['run_type'] == 'train':
        env = VisdomMonitor(env,
                            directory = config.VIDEO_DIR,
                            video_callable = lambda x : x % config.VIS_INTERVAL == 0,
                            uid = str(rank)
                            )
    else:
        env = EvalEnvWrapper(env,
                            directory=config.VIDEO_DIR,
                            uid=str(rank)
                            )
    return env


def construct_envs(config,env_class, mode='vectorenv', make_env_fn=make_env_fn, run_type='train', no_val=False):
    num_processes, num_val_processes = config.NUM_PROCESSES, config.NUM_VAL_PROCESSES
    total_num_processes = num_processes + num_val_processes
    if no_val: num_val_processes = 0
    configs = []
    env_classes = [env_class for _ in range(total_num_processes)]

    # for debug!
    # config.defrost()
    # print('***!!!!!!!!!!!!!!!!**************debug code not deleted')
    # config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['S9hNv5qa7GM','B6ByNegPMKs']
    # config.freeze()

    habitat_api_path = os.path.join(os.path.dirname(habitat.__file__), '../')
    config.defrost()
    config.TASK_CONFIG.DATASET.SCENES_DIR = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.SCENES_DIR)
    config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(habitat_api_path, config.TASK_CONFIG.DATASET.DATA_PATH)
    config.freeze()

    eval_config = config.clone()
    eval_config.defrost()
    eval_config.TASK_CONFIG.DATASET.SPLIT = 'val'
    eval_config.freeze()

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    training_scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        training_scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
        eval_scenes = dataset.get_scenes_to_load(eval_config.TASK_CONFIG.DATASET)

    else:
        eval_scenes = ['EU6Fwq7SyZv']
    if num_processes > 1:
        if len(training_scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        if len(training_scenes) < num_processes:
            raise RuntimeError(
                "reduce the number of processes as there "
                "aren't enough number of scenes"
            )

    random.shuffle(training_scenes)

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(training_scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    eval_scene_splits = [[] for _ in range(num_val_processes)]
    if num_val_processes > 0 :
        for idx, scene in enumerate(eval_scenes):
            eval_scene_splits[idx % len(eval_scene_splits)].append(scene)
    else:
        eval_scenes = []

    scene_splits += eval_scene_splits
    print('Total Process %d = train %d + eval %d '%(total_num_processes, num_processes, num_val_processes))
    for i, s in enumerate(scene_splits):
        if i < num_processes:
            print('train_proc %d :'%i, s)
        else:
            print('eval_proc %d :' % i, s)

    assert sum(map(len, scene_splits)) == len(training_scenes+eval_scenes)

    for i in range(total_num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.DATASET.SPLIT = 'train' if i < num_processes else 'val'
        if len(training_scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        #task_config = add_panoramic_camera(task_config)

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )

        proc_config.freeze()
        configs.append(proc_config)

    if mode == 'vectorenv':
        envs = habitat.VectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(
                tuple(zip(configs, env_classes, range(total_num_processes), [{'run_type':run_type}]*total_num_processes))
            )
        )
    else:
        envs = make_env_fn(configs[0] ,env_class, 0, { 'run_type': run_type})
    return envs
