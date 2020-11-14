#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
import os

DEFAULT_CONFIG_DIR = "IL_configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.VERSION = 'base'
_C.AGENT_TASK = 'search'
_C.BASE_TASK_CONFIG_PATH = "IL_configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "logs/"
_C.VIDEO_DIR = "data/video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/eval_checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.NUM_VAL_PROCESSES = 0
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/new_checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.VIS_INTERVAL = 200
_C.DIFFICULTY = 'easy'
_C.NUM_GOALS = 1
_C.REWARD_METHOD = 'progress'
_C.POLICY = 'PointNavResNetPolicy'
_C.OBS_TO_SAVE = ['panoramic_rgb', 'panoramic_depth', 'target_goal']
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "test"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.SUCCESS_DISTANCE = 1.0
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.rnn_type = "LSTM"
_C.RL.PPO.num_recurrent_layers = 2
_C.RL.PPO.backbone = "resnet50"
_C.RL.PPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
_C.RL.PPO.pretrained = False
_C.RL.PPO.il_pretrained = False
_C.RL.PPO.pretrained_encoder = False
_C.RL.PPO.train_encoder = True
_C.RL.PPO.reset_critic = True

#----------------------------------------------------------------------------
# Base architecture config
_C.features = CN()
_C.features.visual_feature_dim = 512
_C.features.action_feature_dim = 32
_C.features.time_dim = 8

#----------------------------------------------------------------------------
# Transformer
#----------------------------------------------------------------------------
_C.attention = CN()
_C.attention.n_head = 4
_C.attention.d_model = 512 + 32
_C.attention.d_k = 512 + 32
_C.attention.d_v = 512 + 32
_C.attention.dropout = 0.1

# for memory module
_C.memory = CN()
_C.memory.embedding_size = 512
_C.memory.memory_size = 100
_C.memory.pose_dim = 5
_C.memory.need_local_memory = False

_C.training = CN()
_C.training.batch_size = 20
_C.training.lr = 1e-4
_C.training.max_epoch = 100
_C.training.lr_decay = 0.5
_C.training.num_workers = 7

_C.saving = CN()
_C.saving.name = '0715_lgmt'
_C.saving.log_interval = 100
_C.saving.save_interval = 500
_C.saving.eval_interval = 500

_C.dataset = CN()
_C.dataset.max_demo_length = 50
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, config.VERSION)
    config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, config.VERSION)
    config.EVAL_CKPT_PATH_DIR = os.path.join(config.EVAL_CKPT_PATH_DIR, config.VERSION)
    config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, config.VERSION)

    config.freeze()
    return config
