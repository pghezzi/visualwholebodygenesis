import argparse
import numpy as np
import os
from datetime import datetime

import genesis as gs
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning')
    # print info
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(f"Start training for task: {args.task}")
    print(f"num_envs: {env_cfg.env.num_envs}")
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    #    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.offline = True
        args.num_envs = 1
    train(args)
