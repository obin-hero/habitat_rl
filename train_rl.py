#!/usr/bin/env python3

# Copyright (without_goal+curr_emb) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import random

import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from rl_configs.default import get_config

from trainer.algo import ppo, ddppo
import env_utils
import os
os.environ['GLOG_minloglevel'] = "2"
os.environ['MAGNUM_LOG'] = "quiet"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-type",
    choices=["train", "eval", 'benchmark'],
    required=True,
    help="run type of the experiment (train or eval)",
)
parser.add_argument(
    "--exp-config",
    type=str,
    required=True,
    help="path to config yaml containing info about experiment",
)
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    help="gpus",
)
parser.add_argument(
    "--stop",
    action='store_true',
    default=False,
    help="include stop action or not",
)
parser.add_argument(
    "--diff",
    choices=['easy', 'medium', 'hard'],
    help="episode difficulty",
)
parser.add_argument(
    "--seed",
    type=str,
    default="none"
)
arguments = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
# print(args.gpu)



def main():
    run_exp(**vars(arguments))


def run_exp(exp_config: str, run_type: str, opts=None, *args, **kwargs) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()
    config.DIFFICULTY = arguments.diff
    if arguments.stop:
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    else:
        config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    if arguments.seed != 'none':
        config.TASK_CONFIG.SEED = int(arguments.seed)
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED) 
    np.random.seed(config.TASK_CONFIG.SEED)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == 'benchmark':
        trainer.benchmark()


if __name__ == "__main__":
    main()
