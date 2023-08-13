import os
from datetime import datetime
run_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
from PPO.PPO import PPO
import numpy as np
from collections import deque

from AlienEnv.alienrl_env import AlienRLEnv

env_name = "AlienRLEnv"

env = AlienRLEnv()

# Not implemented in current version, currently, batch_size = buffer_size
batch_size = 512

max_training_timesteps = 1_000_000  # break training loop if timesteps > max_training_timesteps

print_freq = 10 # batch_size * 10
save_model_freq = 50_000

# Starting standard deviation for action distribution
action_sd = 0.6
# Linearly decay action_sd where, action_sd = action_sd - action_sd_decay_rate
action_sd_decay_rate = 0.05        
# Set minimum action standard deviation
min_action_sd = 0.1                
# action standard devation decay frequency
action_sd_decay_freq = 250000

# Batch/buffer size for training, should be multiple of batch_size
# buffer_size = batch_size * 1  # 1024 - Converged faster, at 300k timesteps (ent_coef = 0.0)
buffer_size = batch_size * 4  # 4096 - Converged at 500k timesteps (ent_coef = 0.001)
# buffer_size = batch_size * 40 # 40960 - Converges at much slower rate and stable rate

# Update policy for n epochs
num_of_epochs = 128 # 80

eps_clip = 0.2
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001
ent_coef = 0.0 # 0.001 # Increasing entropy coefficient helps exploration, 0 seems to be the best value
vf_coef = 0.5

state_dim = sum(env.observation_space['framestack'].shape) + env.observation_space['telemetry'].shape[0]

# action space dimension
action_dim = env.action_space.shape[0]

checkpoint_path = "models" + '/' + f"{run_start_time}_{ent_coef}" + "/"

if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

logs_dir = f"runs/{run_start_time}_{ent_coef}"

writer = SummaryWriter(logs_dir)

# initialize a PPO agent
agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, num_of_epochs, eps_clip, ent_coef, vf_coef, action_sd)

print("Initialisation complete.")

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Training started at: ", start_time)

# printing and logging variables
total_episodes = 0

global_step_num = 0
episode_num = 1

reward_history = deque(maxlen=100)
best_reward = env.reward_range[0]

# training loop
while global_step_num <= max_training_timesteps:

    state, _ = env.reset()
    episode_reward = 0
    done = False
    trunc = False

    while not done and not trunc:
        
        # Select action with policy
        action = agent.select_action(state)
        state, reward, done, trunc, _ = env.step(action)

        # Saving reward and is_terminals
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done or trunc)

        global_step_num += 1
        episode_reward += reward

        # Update agent
        if global_step_num % buffer_size == 0:
            print("Updating agent...")
            agent.update()

        # Decay action std of ouput action distribution
        if global_step_num % action_sd_decay_freq == 0:
            agent.decay_action_sd(action_sd_decay_rate, min_action_sd)

        if global_step_num % save_model_freq == 0:
            print("Saving model...")
            agent.save(f"{checkpoint_path}{global_step_num}.pth")

    reward_history.append(episode_reward)
    avg_reward = np.mean(reward_history)

    if avg_reward > best_reward and len(reward_history) >= 100:
        best_reward = avg_reward
        
    
    if (total_episodes+1) % print_freq == 0:
        print(f"Episode: {episode_num} \t Total Steps: {global_step_num} \t Average Reward: {avg_reward:9.02f} \t Best Reward: {best_reward:.02f}, \t Elapsed Time: {datetime.now().replace(microsecond=0) - start_time}")

    writer.add_scalar('Reward', episode_reward, global_step=global_step_num)
    writer.add_scalar('Average Reward', avg_reward, global_step=global_step_num)

    total_episodes += 1

    episode_num += 1

writer.close()
env.close()

end_time = datetime.now().replace(microsecond=0)
print()
print("Started training at: ", start_time)
print("Finished training at: ", end_time)
print("Total training time: ", end_time - start_time)