import torch
import torch.nn as nn
from PPO.agent import ActorCritic
from PPO.memory import PPOMemory

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, num_of_epochs, eps_clip, ent_coef, vf_coef, action_sd_init=0.6):

        self.action_sd = action_sd_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.num_of_epochs = num_of_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        self.buffer = PPOMemory()

        self.policy = ActorCritic(state_dim, action_dim, action_sd_init, device).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_sd_init, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.prev_action = None
        self.max_delta_action = 0.01

    def set_action_sd(self, new_action_sd):
        self.action_sd = new_action_sd
        self.policy.set_action_sd(new_action_sd)
        self.policy_old.set_action_sd(new_action_sd)

    # Hones in on optimal policy by reducing action standard deviation
    def decay_action_sd(self, action_sd_decay_rate, min_action_sd):
        # breakpoint()
        self.action_sd = self.action_sd - action_sd_decay_rate
        self.action_sd = round(self.action_sd, 4)
        if (self.action_sd <= min_action_sd):
            self.action_sd = min_action_sd
            print("Setting actor output action standard deviation to minimum standard deviation: ", self.action_sd)
        else:
            print("Setting actor output action standard deviation to: ", self.action_sd)
        self.set_action_sd(self.action_sd)

    def select_action(self, state):

        with torch.no_grad():

            fs = torch.FloatTensor(state['framestack']).unsqueeze(0).to(device)
            # print(fs.shape)
            # fs = fs.view(1,3,84,84)

            tel = torch.FloatTensor(state['telemetry']).unsqueeze(0).to(device)
            # print(tel.shape)
            # tel = tel.view(1, -1)
            
            state = {'framestack': fs, 'telemetry': tel}
        
            action, action_logprob, state_val = self.policy_old.act(state)

        # Clip action to not change more than 0.1 from previous action
        # if self.prev_action is not None:
        #     action = torch.clip(action, self.prev_action - self.max_delta_action, self.prev_action + self.max_delta_action)
        #     self.prev_action = action

        # if self.prev_action is None:
        #     self.prev_action = action

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # breakpoint()
        # assuming self.buffer.states is a list of dictionaries
        framestack_states = torch.stack([state['framestack'] for state in self.buffer.states], dim=0)
        telemetry_states = torch.stack([state['telemetry'] for state in self.buffer.states], dim=0)
        
        # apply squeeze, detach, and device transfer operations if necessary
        framestack_states = torch.squeeze(framestack_states).detach().to(device)
        telemetry_states = torch.squeeze(telemetry_states).detach().to(device)
        
        # now old_states is a dictionary
        old_states = {'framestack': framestack_states, 'telemetry': telemetry_states}
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # breakpoint()
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for n number of epochs
        for _ in range(self.num_of_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surrogate_loss_1 = ratios * advantages
            surrogate_loss_2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            value_loss = self.MseLoss(state_values, rewards)

            # final loss of clipped objective PPO
            loss = -torch.min(surrogate_loss_1, surrogate_loss_2) + self.vf_coef * value_loss - self.ent_coef * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


