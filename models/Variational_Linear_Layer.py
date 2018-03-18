import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

class Variational_Linear_Layer():
    def __init__(self, d_in, d_out, rho, dtype=FloatTensor, bias=True):
        self.d_in = d_in
        self.d_out = d_out
        self.rho = rho
        self.bias = bias
       
        self.W_mu = Variable(torch.Tensor(d_out, d_in).type(dtype), requires_grad=True)
        self.W_rho = Variable(torch.Tensor(self.W_mu.size()).type(dtype), requires_grad=True)
        if self.bias:
            self.b_mu = Variable(torch.Tensor(d_out).type(dtype), requires_grad=True)
            self.b_rho = Variable(torch.Tensor(self.b_mu.size()).type(dtype), requires_grad=True)
      
        self.reset_parameters() #initializes W and b

        self.W_sample = None
        if self.bias:
            self.b_sample = None
        
        if self.bias:
            self.mu_l = [self.W_mu, self.b_mu]
            self.rho_l = [self.W_rho, self.b_rho]
        else:
            self.mu_l = [self.W_mu]
            self.rho_l = [self.W_rho]

    # given rho, calculate standard deviation (sigma)
    @staticmethod
    def calc_sigma(rho):
        return torch.log(1.0 + torch.exp(rho))

    def get_mu_l(self):
        return self.mu_l

    def get_rho_l(self):
        return self.rho_l

    def get_sigma_l(self):
        return [Variational_Linear_Layer.calc_sigma(rho) for rho in self.rho_l]

    def parameters(self):
        return self.get_mu_l() + self.get_rho_l()

    def reset_parameters(self):
        self.W_mu.data.fill_(0.0)
        # torch.nn.init.xavier_uniform(self.W_mu.data)
        self.W_rho.data.fill_(self.rho)

        if self.bias:
            self.b_mu.data.fill_(0.0)
            self.b_rho.data.fill_(self.rho)

    def make_target(self):
        target = Variational_Linear_Layer(self.d_in, self.d_out, self.rho, bias=self.bias)
        target.W_mu = Variable(self.W_mu.data.clone(), requires_grad=False)
        if self.bias:
            target.b_mu = Variable(self.b_mu.data.clone(), requires_grad=False)
            target.mu_l = [target.W_mu, target.b_mu]
        else:
            target.mu_l = [target.W_mu]
        return target

    #Use the current distribution parameters over W and b to sample W and b.
    #Saves the sample Variable in self.W_sample and self.b_sample, and returns the samples
    def sample(self):

        W_sigma = Variational_Linear_Layer.calc_sigma(self.W_rho)
        W_sample = self.W_mu + Variable(torch.randn(self.W_mu.size()), requires_grad=True) * W_sigma
        self.W_sample = W_sample

        if self.bias:
            b_sigma = Variational_Linear_Layer.calc_sigma(self.b_rho)
            b_sample = self.b_mu + Variable(torch.randn(self.b_mu.size()), requires_grad=True) * b_sigma
            self.b_sample = b_sample
            return [self.W_sample, self.b_sample]
        else: 
            return [self.W_sample]

    #calculate x*W_sample + b_sample, and return the Variable
    def forward(self, x, mean_only):
        if mean_only:
            if self.bias:
                return F.linear(x, self.W_mu, self.b_mu)
            else:
                return F.linear(x, self.W_mu)
        if self.W_sample is None or (self.bias and self.b_sample is None):
            raise Exception("Must sample W and b before calling forward")
        if self.bias:
            return F.linear(x, self.W_sample, self.b_sample)
        else:
            return F.linear(x, self.W_sample)
