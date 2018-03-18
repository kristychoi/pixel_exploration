import torch
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# Debug/display helper functions
def get_Q(model, state):
    var = Variable(state, volatile=True).type(FloatTensor)
    if model.variational():
        if model.target is not None:
            return model.target_q(var).data
        return model(var, mean_only=True).data
    else:
        return model(var).data


def Q_values(env, model):
    n = env.state_size()
    # n = env.observation_space.n
    states = np.identity(n)
    Q = torch.zeros(n, env.num_actions())
    for i, row in enumerate(states):
        state = Tensor(row).unsqueeze(0)
        Q[i] = get_Q(model, state)[0]
    return Q


def start_state_value(env, model):
    start = Tensor(env.get_start_state()).unsqueeze(0)
    Q = get_Q(model, start)
    return torch.max(Q)


def Q_dump(env, model):
    n = env.state_size()
    # n = env.observation_space.n
    m = int(n ** 0.5)
    Q = Q_values(env, model)
    for i, row in enumerate(Q.t()):
        print("Action {}".format(i))
        print(row.contiguous().view(m, m))
        # print(row.contiguous())