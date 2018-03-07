# -*- coding: utf-8 -*-

import gym
import gym_gridworld

import math
import random
from collections import namedtuple, defaultdict
from time import time

import torch.optim as optim
from models.linear_models import *
from utils.helpers import *
from configs.grid_config import Config
from utils.updated_replay import ReplayBuffer
######################################

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

effective_eps = 0.0  # for printing purposes


def select_action(env, model, state, steps_done):
    if model.variational():
        var = Variable(state, volatile=True).type(FloatTensor)
        q_sa = model(var).data
        best_action = q_sa.max(1)[1]
        return LongTensor([best_action[0]]).view(1, 1)

    sample = random.random()

    def calc_ep(start, end, decay, t):
        if config.linear_decay:
            return start - (float(min(t, decay)) / decay) * (start - end)
        else:
            return end + (start - end) * math.exp(-1. * t / decay)

    eps_threshold = calc_ep(config.ep_start, config.ep_end, config.ep_decay, steps_done)

    global effective_eps
    effective_eps = eps_threshold
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[
            1].view(1, 1)
    else:
        return LongTensor([[random.randrange(env.num_actions())]])


#########################################################################################
def simulate(model, env, config):
    optimizer = optim.Adam(model.parameters())
    memory = ReplayBuffer(config.replay_mem_size, config.frame_history_len)

    last_sync = [0]  # in an array since py2.7 does not have "nonlocal"

    loss_list = []
    sigma_average_dict = defaultdict(list)

    def optimize_model(model):
        if not memory.can_sample(config.batch_size):
            return

        def loss_of_batch(batch):
            # convert numpy sampled items into Variable torch Tensors
            # todo: will need to add mc_batch here later for MMC update
            state_batch = Variable(torch.from_numpy(batch[0]).type(FloatTensor))
            action_batch = Variable(torch.from_numpy(batch[1]).type(
                LongTensor)).unsqueeze(1)
            reward_batch = Variable(torch.from_numpy(batch[2]))
            next_states = Variable(torch.from_numpy(batch[3]).type(FloatTensor),
                                   volatile=True)
            not_done_mask = Variable(torch.from_numpy(1-batch[4]).type(FloatTensor),
                                     volatile=True)

            # compute bonuses here if necessary
            # if config.bonus:
            #     states_visited = np.nonzero(state_batch.data.numpy())[1]
            #     bonus_batch = config.beta/np.sqrt(memory.count_table[states_visited])
            #     aug_reward = reward_batch.data.numpy() + bonus_batch
            #     # clip rewards if necessary
            #     # aug_reward = np.maximum(-1.0, np.minimum(aug_reward, 1.0))
            #     rew_batch = Variable(FloatTensor(aug_reward))

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            state_action_values = model(state_batch).gather(1, action_batch).view(-1)

            # Compute V(s_{t+1}) for all next states.
            expected_state_action_values = model.target_value(reward_batch, config.gamma,
                                                              next_states, not_done_mask)

            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            expected_state_action_values.volatile = False

            loss = (state_action_values - expected_state_action_values).pow(2).sum()

            # if model.variational():
            #     loss += loss_of_sample()

            return loss

        def optimizer_step(transitions):
            # compute loss
            loss = loss_of_batch(transitions)
            loss_list.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

        if config.train_in_epochs:
            M = int(memory.num_in_buffer / config.batch_size)
            for target_iter in range(config.num_target_reset):
                model.save_target()
                for epoch in range(config.num_epochs):
                    for minibatch in range(M):
                        transitions = memory.sample(config.batch_size)
                        optimizer_step(transitions)
        else:
            if last_sync[0] % config.period_target_reset == 0:
                model.save_target()
                print("Target reset")
            last_sync[0] += 1
            transitions = memory.sample(config.batch_size)
            optimizer_step(transitions)

    time_list = []
    value_list = []
    score_list = []

    start_time = time()
    i_episode = 0
    steps_done = 0

    while time() - start_time < config.train_time_seconds:
        # Initialize the environment and state
        state = env.reset()
        iters = 0
        score = 0
        while iters < config.max_ep_len:
            do_update = False
            if iters % config.period_sample == 0:
                # if model.variational():
                #     w_sample = model.sample()
                do_update = not config.train_in_epochs
            iters += 1

            # this stores the state in the replay buffer as a numpy array or (9,)
            idx = memory.store_frame(state)
            q_input = Tensor(memory.encode_recent_observation()).unsqueeze(0)

            # Select and perform an action
            action = select_action(env, model, q_input, steps_done)
            steps_done += 1
            next_state, reward, done, _ = env.step(action[0, 0])
            score += reward

            # Store the transition in memory
            memory.store_effect(idx, action[0,0], reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if do_update:
                optimize_model(model)
            if done:
                break

        # log results
        if i_episode % 100 == 0:
            if model.variational():
                print("Episode: {}\tscore: {}".format(i_episode, score))
            else:
                print("Episode: {}\tscore: {}\tepsilon: {}".format(i_episode, score,
                                                                   effective_eps))

        value = start_state_value(env, model)
        elapsed = time() - start_time

        time_list.append(elapsed)
        value_list.append(value)
        score_list.append(score)

        if config.train_in_epochs and i_episode % config.period_train_in_epochs == 0:
            optimize_model(model)
        i_episode += 1

    # print(memory.state_action_counts())
    Q_dump(env, model)
    return loss_list, score_list, time_list, value_list, sigma_average_dict['W']


if __name__ == '__main__':
    # set seeds
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # grab config file
    config = Config()

    env = gym.make(config.env_name).unwrapped
    models = []

    models.append(lambda: ("DQN", Linear_DQN(env.state_size(), env.num_actions())))

    color_dict = {"DQN": 'red', "Double DQN": "green", "BBQN": "blue", "Heavy BBQN": "yellow"}

    for i, constructor in enumerate(models):
        name, model = constructor()
        loss_average, score_list, time_list, value_list, sigma_average = simulate(
            model, env, config)