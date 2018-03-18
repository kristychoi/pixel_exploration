import gym
import gym.spaces
import itertools

import sys
import random
from collections import namedtuple

import torch
import numpy as np
from copy import deepcopy
from utils.updated_replay import ReplayBuffer, MMCReplayBuffer
from utils.schedule import LinearSchedule
from utils.gym_atari_wrappers import get_wrapper_by_name
from torch.autograd import Variable
import pickle
import logging


OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
    "episode_rewards": []
}


# use GPU if available
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


def dqn_learn(env, q_func, optimizer_spec, density, cnn_kwargs, config,
              exploration=LinearSchedule(1000000, 0.1), stopping_criterion=None):
    """
    Run Deep Q-learning algorithm.
    """
    # this is just to make sure that you're operating in the correct environment
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, config.frame_history_len * img_c)
    num_actions = env.action_space.n

    # define Q network and target network (instantiate 2 DQN's)
    in_channel = input_shape[-1]
    Q = q_func(in_channel, num_actions)
    target_Q = deepcopy(Q)

    # call tensorflow wrapper to get density model
    if config.bonus:
        pixel_bonus = density(FLAGS=cnn_kwargs)

    if USE_CUDA:
        Q.cuda()
        target_Q.cuda()

    # define eps-greedy exploration strategy
    def select_action(model, obs, t):
        """
        Selects random action w prob eps; otherwise returns best action
        :param exploration:
        :param t:
        :return:
        """
        if config.egreedy_exploration:
            sample = random.random()
            eps_threshold = exploration.value(t)
            if sample > eps_threshold:
                obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0) / 255.0
                return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
            else:
                # this returns a number
                return LongTensor([[random.randrange(num_actions)]])[0,0]
        # no exploration; just take best action
        else:
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0) / 255.0
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
    # construct torch optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # construct the replay buffer
    if config.mmc:
        replay_buffer = MMCReplayBuffer(config.replay_buffer_size, config.frame_history_len)
    else:
        replay_buffer = ReplayBuffer(config.replay_buffer_size, config.frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()

    # index trackers for updating mc returns
    episode_indices_in_buffer = []
    reward_each_timestep = []
    timesteps_in_buffer = []
    cur_timestep = 0

    # monte_carlo returns
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # process last_obs to include context from previous frame
        last_idx = replay_buffer.store_frame(last_obs)

        # record where this is in the buffer
        episode_indices_in_buffer.append(last_idx)
        timesteps_in_buffer.append(cur_timestep)
        # one more step in episode
        cur_timestep += 1

        # take latest observation pushed into buffer and compute corresponding input
        # that should be given to a Q network by appending some previous frames
        recent_obs = replay_buffer.encode_recent_observation()
        # recent_obs.shape is also (84, 84, 4)

        # choose random action if not yet started learning
        if t > config.learning_starts:
            action = select_action(Q, recent_obs, t)
        else:
            action = random.randrange(num_actions)

        # advance one step
        obs, reward, done, _ = env.step(action)
        # clip reward to be in [-1, +1]
        reward = max(-1.0, min(reward, 1.0))

        ###############################################
        # do density model stuff here
        if config.bonus:
            intrinsic_reward = pixel_bonus.bonus(obs, t)
            if t % config.log_freq == 0:
                logging.info('t: {}\t intrinsic reward: {}'.format(t, intrinsic_reward))

            # add intrinsic reward to clipped reward
            reward += intrinsic_reward
            # clip reward to be in [-1, +1] once again
            reward = max(-1.0, min(reward, 1.0))
            assert -1.0 <= reward <= 1.0
        ################################################

        # store reward in list to use for calculating MMC update
        reward_each_timestep.append(reward)
        replay_buffer.store_effect(last_idx, action, reward, done)

        # reset environment when reaching episode boundary
        if done:
            # only if computing MC return
            if config.mmc:
                # episode has terminated --> need to do MMC update here
                # loop through all transitions of this past episode and add in mc_returns
                assert len(timesteps_in_buffer) == len(reward_each_timestep)
                mc_returns = np.zeros(len(timesteps_in_buffer))

                # compute mc returns
                r = 0
                for i in reversed(range(len(mc_returns))):
                    r = reward_each_timestep[i] + config.gamma * r
                    mc_returns[i] = r

                # populate replay buffer
                for j in range(len(mc_returns)):
                    # get transition tuple in reward buffer and update
                    update_idx = episode_indices_in_buffer[j]
                    # put mmc return back into replay buffer
                    replay_buffer.mc_return_t[update_idx] = mc_returns[j]
            # reset because end of episode
            episode_indices_in_buffer = []
            timesteps_in_buffer = []
            cur_timestep = 0
            reward_each_timestep = []

            # reset
            obs = env.reset()
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken

        # perform training
        if (t > config.learning_starts and t % config.learning_freq == 0 and
                replay_buffer.can_sample(config.batch_size)):

            # sample batch of transitions --> also grab MMC batch if computing MMC return
            if config.mmc:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, mc_batch = \
                replay_buffer.sample(config.batch_size)
                mc_batch = Variable(torch.from_numpy(mc_batch).type(FloatTensor))
            else:
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = \
                replay_buffer.sample(config.batch_size)

            # convert variables to torch tensor variables
            # (32, 84, 84, 4)
            obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor)/255.0)
            act_batch = Variable(torch.from_numpy(act_batch).type(LongTensor))
            rew_batch = Variable(torch.from_numpy(rew_batch).type(FloatTensor))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(FloatTensor)/255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(FloatTensor))

            # 3.c: train the model: perform gradient step and update the network
            # this returns [32, 18] --> [32 x 1]
            # i squeezed this so that it'll give me [32]
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()

            # goes from [32, 18] --> [32]
            # this gives you a FloatTensor of size 32 // gives values of max
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]

            # torch.FloatTensor of size 32
            next_Q_values = not_done_mask * next_max_q

            # this is [r(x,a) + gamma * max_a' Q(x', a')]
            target_Q_values = rew_batch + (config.gamma * next_Q_values)

            if config.mmc:
                mixed_target_Q_values = ((1-config.beta) * target_Q_values) + \
                                        (config.beta * mc_batch)
                # replace target_Q_values with mixed target
                bellman_err = mixed_target_Q_values - current_Q_values
            else:
                bellman_err = target_Q_values - current_Q_values

            # clip gradient
            clipped_bellman_err = bellman_err.clamp(-1, 1)

            d_err = clipped_bellman_err * -1.0
            optimizer.zero_grad()

            # design decision will affect this backward propagation
            current_Q_values.backward(d_err.data)
            # current_Q_values.backward(d_err.data.unsqueeze(1))

            # perform param update
            optimizer.step()
            num_param_updates += 1

            # periodically update the target network
            if num_param_updates % config.target_update_freq == 0:
                target_Q = deepcopy(Q)

            ### 4. Log progress
            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            # save statistics
            Statistic["mean_episode_rewards"].append(mean_episode_reward)
            Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
            Statistic["episode_rewards"].append(episode_rewards)

            if t % config.log_freq == 0 and t > config.learning_starts:
                logging.info("Timestep %d" % (t,))
                logging.info("mean reward (100 episodes) %f" % mean_episode_reward)
                logging.info("best mean reward %f" % best_mean_episode_reward)
                logging.info("episodes %d" % len(episode_rewards))
                logging.info("exploration %f" % exploration.value(t))
                sys.stdout.flush()

                # Dump statistics to pickle
            if t % 1000000 == 0 and t > config.learning_starts:
                with open(config.output_path + 'statistics.pkl', 'wb') as f:
                    pickle.dump(Statistic, f)


def test_dqn_learning(env, q_func, optimizer_spec, density, cnn_kwargs, config,
          exploration=LinearSchedule(1000000, 0.1), stopping_criterion=None):

    """Run Deep Q-learning algorithm."""
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = config.frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(FloatTensor).unsqueeze(0) / 255.0
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(FloatTensor)
    target_Q = q_func(input_arg, num_actions).type(FloatTensor)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size, config.frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000

    for t in itertools.count():
        ### Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        last_idx = replay_buffer.store_frame(last_obs)
        # encode_recent_observation will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        recent_observations = replay_buffer.encode_recent_observation()

        # Choose random action if not yet start learning
        if t > config.learning_starts:
            action = select_epilson_greedy_action(Q, recent_observations, t)[0, 0]
        else:
            action = random.randrange(num_actions)
        # Advance one step
        obs, reward, done, _ = env.step(action)
        # clip rewards between -1 and 1
        reward = max(-1.0, min(reward, 1.0))
        # Store other info in replay memory
        replay_buffer.store_effect(last_idx, action, reward, done)
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > config.learning_starts and
                t % config.learning_freq == 0 and
                replay_buffer.can_sample(config.batch_size)):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(
                config.batch_size)
            # Convert numpy nd_array to torch variables for calculation
            obs_batch = Variable(torch.from_numpy(obs_batch).type(FloatTensor) /
                                 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(
                torch.from_numpy(next_obs_batch).type(FloatTensor) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(FloatTensor)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1))
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            next_max_q = target_Q(next_obs_batch).detach().max(1)[0]
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (config.gamma * next_Q_values)
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values
            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            current_Q_values.backward(d_error.data.unsqueeze(1))

            # Perfom the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % config.target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward,
                                           mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > config.learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()