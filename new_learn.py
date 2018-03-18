import gym
import gym.spaces
import itertools

import sys
import random
from collections import namedtuple

from models.linear_models import *
from utils.helpers import *
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
USE_CUDA = True
# USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor


def atari_learn(env,
                q_func,
                config,
                optimizer_spec,
                exploration=LinearSchedule(1000000, 0.1),
                stopping_criterion=None):
    """Run Deep Q-learning algorithm."""
    ###################################
    # # todo: just for easy checking
    # seed = 1234
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # config = Config()
    # env = get_env(config.env_name, seed, config.downsample)
    # q_func = DQN
    # def stopping_criterion(env, t):
    #     # t := num steps of wrapped env // different from num steps in underlying env
    #     return get_wrapper_by_name(env, "Monitor").get_total_steps() >= \
    #            config.max_timesteps
    #
    # # decay schedule
    # exploration = LinearSchedule(1000000, 0.1)
    ###################################

    # check to make sure that we're operating in the correct environment
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
    in_channel = input_shape[-1]

    # define Q network and target network (instantiate 2 DQN's)
    Q = q_func(in_channel, num_actions)
    target_Q = q_func(in_channel, num_actions)

    # if GPU enabled
    if USE_CUDA:
        Q.cuda()
        target_Q.cuda()

    ######
    # epsilon-greedy exploration
    def select_action(model, state, t):
        state = torch.from_numpy(state).type(FloatTensor).unsqueeze(0) / 255.0
        # if no exploration, just return
        if not Q.random_exploration():
            var = Variable(state, volatile=True)
            q_sa = model(var).data
            best_action = q_sa.max(1)[1]
            return LongTensor([best_action[0]]).view(1,1)
        else:
            # epsilon-greedy exploration
            sample = random.random()
            eps_threshold = exploration.value(t)
            if sample > eps_threshold:
                return model(Variable(state, volatile=True)).data.max(1)[1].view(1,1)
            else:
                return LongTensor([[random.randrange(num_actions)]])
    ######
    # define optimizer
    # optimizer = torch.optim.Adam(Q.parameters())
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(config.replay_mem_size, config.frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    loss_list = []
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    state = env.reset()

    # step through environment
    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        idx = replay_buffer.store_frame(state)
        q_input = replay_buffer.encode_recent_observation()

        # select action
        action = select_action(Q, q_input, t)[0, 0]

        # take action
        next_state, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0))  # clip rewards

        # store transition in memory
        replay_buffer.store_effect(idx, action, reward, done)

        # reset environment if end of episode
        if done:
            next_state = env.reset()
        # move onto next state
        state = next_state
        #####

        ### 3. Perform experience replay and train the network.
        if (t > config.learning_starts and
                t % config.learning_freq == 0 and replay_buffer.can_sample(
                    config.batch_size)):
            #####
            # optimize model
            state_batch, action_batch, reward_batch, next_state_batch, done_mask = \
                replay_buffer.sample(config.batch_size)

            # turn things into pytorch variables
            state_batch = Variable(torch.from_numpy(state_batch).type(
                FloatTensor) / 255.0)
            action_batch = Variable(torch.from_numpy(action_batch).type(
                LongTensor))
            reward_batch = Variable(torch.from_numpy(reward_batch).type(FloatTensor))
            next_state_batch = Variable(torch.from_numpy(next_state_batch).type(
                FloatTensor) / 255.0, volatile=True)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(FloatTensor),
                                     volatile=True)
            # next_state_batch = Variable(torch.from_numpy(next_state_batch).type(
            #     FloatTensor) / 255.0)
            # not_done_mask = Variable(torch.from_numpy(1 - done_mask).type(FloatTensor))

            # compute Q values
            # todo: check shape
            state_action_values = Q(state_batch).gather(
                1, action_batch.unsqueeze(1)).view(-1)  # ([32])

            # compute target Q values
            next_max_q = target_Q(next_state_batch).max(1)[0]  # ([32])
            # next_max_q = target_Q(next_state_batch).detach().max(1)[0]  # ([32])
            expected_state_action_values = reward_batch + (config.gamma * not_done_mask
                                                           * next_max_q)  # ([32])

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            expected_state_action_values.volatile = False

            # todo fix
            bellman_err = expected_state_action_values - state_action_values
            clipped_bellman_error = bellman_err.clamp(-1, 1)
            d_error = clipped_bellman_error * -1.0
            state_action_values.backward(d_error.data.unsqueeze(1))

            ############
            # if config.deep:
            #     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            # else:
            #     loss = (state_action_values - expected_state_action_values).pow(2).sum()
            # loss_list.append(loss.data[0])
            # #####
            # optimizer.zero_grad()
            # # pass back gradient
            # loss.backward()
            #
            # # clip gradient
            # if config.clip_grad:
            #     nn.utils.clip_grad_norm(Q.parameters(), 10.)
            #########################

            # take parameter step
            optimizer.step()
            num_param_updates += 1

            # periodically update target network
            if num_param_updates % config.target_update_freq == 0:
                target_Q = deepcopy(Q)

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % config.log_freq == 0:
            logging.info("Timestep %d" % (t,))
            logging.info("mean reward (100 episodes) %f" % mean_episode_reward)
            logging.info("best mean reward %f" % best_mean_episode_reward)
            logging.info("episodes %d" % len(episode_rewards))
            logging.info("exploration %f" % exploration.value(t))
            # logging.info('average loss: %f' % np.mean(loss_list))


def old_learn(env, q_func, optimizer_spec, density, cnn_kwargs, config,
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
        # todo: do density model stuff here
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
                # mixed MC update would be: todo: flipped beta/1-beta from paper
                mixed_target_Q_values = (config.beta * target_Q_values) + (1 - config.beta)* mc_batch
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