"""
Driver file for running DQN agent on Atari
"""
import os
import gym
import numpy as np
import random
import logging
import argparse
from models.linear_models import *
from models.deep_dqn import DQN
import utils
from utils.gym_atari_wrappers import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule
from configs.dqn_config import Config
from learn import OptimizerSpec, old_learn
from utils.tf_wrapper import PixelBonus

# do logging
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stderr))


def print_key_pairs(v, title="Parameters"):
    """
    Print key-value pairs for user-specified args
    ---> borrowed from avast's benchmarks.utils
    :param v:
    :param title:c
    :return:
    """
    items = v.items() if type(v) is dict else v
    logging.info("\n" + "-" * 40)
    logging.info(title)
    logging.info("-" * 40)
    for key,value in items:
        logging.info("{:<20}: {:<10}".format(key, str(value)))
    logging.info("-" * 40)


def update_tf_wrapper_args(args, tf_flags):
    """
    take input command line args to DQN agent and update tensorflow wrapper default
    settings
    :param args:
    :param FLAGS:
    :return:
    """
    # doesn't support boolean arguments
    to_parse = args.wrapper_args
    if to_parse:
        for kwarg in to_parse:
            keyname, val = kwarg.split('=')
            if keyname in ['ckpt_path', 'data_path', 'samples_path', 'summary_path']:
                # if directories don't exist, make them
                if not os.path.exists(val):
                    os.makedirs(val)
                tf_flags.update(keyname, val)
            elif keyname in ['data', 'model']:
                tf_flags.update(keyname, val)
            elif keyname in ['mmc_beta']:
                tf_flags.update(keyname, float(val))
            else:
                tf_flags.update(keyname, int(val))
    return tf_flags


def main(config, env):
    """
    Run DQN on Atari
    :param config:
    :param env:
    :return:
    """
    FLAGS = update_tf_wrapper_args(args, utils.tf_wrapper.FLAGS)

    def stopping_criterion(env, t):
        # t := number of steps of wrapped env
        # different from number of steps in underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= \
               config.max_timesteps

    # optimizer_spec = OptimizerSpec(
    #     constructor=torch.optim.Adam,
    #     kwargs=dict(lr=config.learning_rate, eps=config.epsilon),
    # )
    optimizer_spec = OptimizerSpec(
        constructor=torch.optim.RMSprop,
        kwargs=dict(lr=config.learning_rate, momentum=config.momentum, eps=config.epsilon)
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    old_learn(
        env=env, q_func=DQN, optimizer_spec=optimizer_spec,
        density=PixelBonus, cnn_kwargs=FLAGS, config=config,
        exploration=exploration_schedule, stopping_criterion=stopping_criterion,
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-W", "--wrapper_args", nargs='+',
                           help='args to add onto tensorflow wrapper')
    args = argparser.parse_args()

    # get config file
    config_file = Config()

    # Run training; set seeds for reproducibility
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get environment
    if config_file.deep:
        # this sets up the Atari environment
        env = get_env(config_file.env_name, seed)
    else:
        env = gym.make(config_file.env_name)

    # if directories don't exist, make them
    if not os.path.exists(config_file.output_path):
        os.makedirs(config_file.output_path)

    # Set up logger
    logging.basicConfig(filename=config_file.log_path, level=logging.INFO)

    # print all argument variables
    # print_key_pairs(args.__dict__.items(), title='Command line args')

    main(config_file, env)