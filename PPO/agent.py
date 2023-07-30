import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from PPO.networks import ActorNetwork, CriticNetwork


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_sd_init, device):
        super(ActorCritic, self).__init__()
                
        self.device = device

        self.action_dim = int(action_dim)
        self.action_var = torch.full((self.action_dim,), action_sd_init * action_sd_init).to(self.device)

        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        
    def set_action_sd(self, new_action_sd):
        self.action_var = torch.full((self.action_dim,), new_action_sd * new_action_sd).to(self.device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy