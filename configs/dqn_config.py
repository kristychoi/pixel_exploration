"""
Config file for DQN on Atari 2600 suite
"""


class Config():
    # output config
    output_path = "results/dqn_pong/"
    # todo: actually do model checkpointing and logging
    model_output = output_path + "model.weights/"
    log_path = output_path + "pixel_log.txt"
    plot_output = output_path + "scores.png"

    # environment
    env_name = "PongNoFrameskip-v4"
    deep = True
    clip_grad = True

    # model and train config
    log_freq = 5000
    # save_freq = 5000
    downsample = False  # specific to exploration bonus

    # todo: original hyperparameters; table this for now
    frame_history_len = 4
    replay_buffer_size = 1000000

    max_timesteps = 40000000

    learning_starts = 50000
    batch_size = 32
    target_update_freq = 4
    gamma = 0.99
    learning_freq = 4
    beta = 0.8  # MMC

    # others
    learning_rate = 0.00025
    # lr_multiplier = 1.0
    # alpha = 0.95
    epsilon = 1e-2

    # exploration bonus
    bonus = False