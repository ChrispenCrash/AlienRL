import os
from datetime import datetime

# import logging
run_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# log_file_path = f'logs/{run_start_time}_log.txt'

from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
from PPO.PPO import PPO
import numpy as np
from collections import deque
from time import sleep
import copy
import concurrent.futures
from threading import Lock

from AlienEnv.alienrl_env import AlienRLEnv
from AlienEnv.utils import ActionSmoother

# # Create a custom formatter without the level name
# formatter = logging.Formatter('%(message)s')

# # Configure the root logger to use the custom formatter
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)

# # Create a FileHandler for the log file with the custom formatter
# file_handler = logging.FileHandler(log_file_path)
# file_handler.setFormatter(formatter)
# root_logger.addHandler(file_handler)

# # Create a StreamHandler for the terminal with the custom formatter
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# root_logger.addHandler(stream_handler)

env_name = "AlienRLEnv"

env = AlienRLEnv()

# Not implemented in current version, currently, batch_size = buffer_size
batch_size = 1024

max_training_timesteps = 10_000_000  # break training loop if timesteps > max_training_timesteps

print_freq = 1 # batch_size * 10
save_model_freq = 25_000 #

# Starting standard deviation for action distribution
action_sd = 0.6
# Linearly decay action_sd where, action_sd = action_sd - action_sd_decay_rate
action_sd_decay_rate = 0.05        
# Set minimum action standard deviation
min_action_sd = 0.1                

# action standard devation decay frequency
action_sd_decay_freq = 1_000_000 # 250_000

ent_coef_decay_freq = 1_000_000
ent_coef_decay_rate = 10
min_ent_coef_cutoff = 0.001

# Batch/buffer size for training, should be multiple of batch_size
# buffer_size = batch_size * 1  # 1024 - Converged faster, at 300k timesteps (ent_coef = 0.0)
buffer_size = batch_size * 4  # 4096 - Converged at 500k timesteps (ent_coef = 0.001)
# buffer_size = batch_size * 40 # 40960 - Converges at much slower rate and stable rate

# Update policy for n epochs
num_of_epochs = 128 # 128 # 80 # 10

eps_clip = 0.3
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.0003
ent_coef = 0.01 # 0.001 # 0.001 # Increasing entropy coefficient helps exploration, 0 seems to be the best value
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
agent = PPO(state_dim, action_dim, batch_size, buffer_size, lr_actor, lr_critic, gamma, num_of_epochs, eps_clip, ent_coef, vf_coef, action_sd)

agent.load("best_model.pth")

print("Initialisation complete.")

# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Training started at: ", start_time)

# printing and logging variables
total_episodes = 0

global_step_num = 0
episode_num = 1

best_reward = env.reward_range[0]

# reward_history = deque(maxlen=100)
# episode_times = deque(maxlen=100)
reward_history = []
episode_times = []
agent_update_times = []

def update_agent(agent):
    agent.update()
    return agent

with concurrent.futures.ThreadPoolExecutor() as executor:

    future = None

    smoother = ActionSmoother(alpha=0.8)

    env.controller.set_inputs(0.0,0.5)

    # training loop
    while global_step_num <= max_training_timesteps:

        episode_start_time = datetime.now().replace(microsecond=0)

        
        # Logic to update everytime the car crashes
        # if env.is_hard_reset:
        #     print("Hard reset detected.2")
        #     if len(agent.buffer.rewards) >= buffer_size:
        #         if len(agent.buffer.rewards) > 16384:
        #             agent.buffer.actions = agent.buffer.actions[-16384:]
        #             agent.buffer.rewards = agent.buffer.rewards[-16384:]
        #             agent.buffer.states = agent.buffer.states[-16384:]
        #             agent.buffer.logprobs = agent.buffer.logprobs[-16384:]
        #             agent.buffer.state_values = agent.buffer.state_values[-16384:]
        #             agent.buffer.is_terminals = agent.buffer.is_terminals[-16384:]
        #         # wait for car to crash before updating agent
        #         agent.update()
        #         env.controller.set_inputs(0.0,0.5)
        
        state, _ = env.reset()


        # Potentially add 50% throttle for 2 seconds to get car moving
        # env.controller.set_inputs(0.0,0.5)
        # sleep(2)
        # env.controller.reset_inputs()

        episode_reward = 0
        done = False
        trunc = False

        while not done and not trunc:
            
            # # Collect updated agent
            if future and future.done():
                agent_update_finish_time = datetime.now()
                agent_update_times.append(agent_update_finish_time - agent_update_start_time)
                # t1 = datetime.now()
                # agent.buffer.clear()
                agent = future.result() # replace the old agent with the new, updated one
                # t2 = datetime.now()
                print(f"Agent updated.")
                agent.buffer.clear()
                # t3 = datetime.now()
                print(f"Buffer cleared.")
                # print(f"{len(agent.buffer.states)=}")
                # print(f"{len(agent.buffer.state_values)=}")
                future = None

            # print(len(agent.buffer.states), len(agent.buffer.actions), len(agent.buffer.logprobs),len(agent.buffer.state_values),len(agent.buffer.rewards))
            
            action = agent.select_action(state)
            smoothed_action = smoother.smooth(action)

            state, reward, done, trunc, info = env.step(smoothed_action)

            # with PPO.buffer_lock:
                # Saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done or trunc)

            global_step_num += 1
            episode_reward += reward

            # Update agent
            # if global_step_num % buffer_size == 0:
            if len(agent.buffer.rewards) == buffer_size:
            # if global_step_num % buffer_size == 0:
                if not future: # only create a new future if the old one has finished
                    agent_update_start_time = datetime.now()
                    print("Updating agent...")
                    # t4 = datetime.now()
                    new_agent = copy.deepcopy(agent) # create a copy of the current agent
                    # new_buffer = copy.deepcopy(agent.buffer)
                    # agent.save_weights('temp_weights.pth')
                    # t5 = datetime.now()
                    # print(f"Agent weights saved.")
                    # new_agent = PPO(state_dim, action_dim, batch_size, buffer_size, lr_actor, lr_critic, gamma, num_of_epochs, eps_clip, ent_coef, vf_coef, action_sd)
                    # new_agent.load_weights('temp_weights.pth')
                    t6 = datetime.now()
                    print(f"Agent copied.")
                    
                    # print(f"{len(new_agent.buffer.states)=}")
                    # print(f"{len(new_agent.buffer.state_values)=}")
                    # with PPO.buffer_lock:
                    future = executor.submit(update_agent, new_agent) # update the new agent asynchronously

            # Decay action std of ouput action distribution
            if global_step_num % action_sd_decay_freq == 0:
                agent.decay_action_sd(action_sd_decay_rate, min_action_sd)

            if global_step_num % ent_coef_decay_freq == 0:
                # Should go to 0.001 after 1M steps, 0 after 2M steps
                agent.decay_ent_coef(ent_coef_decay_rate, min_ent_coef_cutoff)

            if global_step_num % save_model_freq == 0:
                print("Saving model...")
                agent.save(f"{checkpoint_path}{global_step_num}.pth")


        episode_times.append(datetime.now().replace(microsecond=0) - episode_start_time)
        reward_history.append(episode_reward)

        avg_episode_time = np.mean(episode_times[-100:]).total_seconds()
        avg_reward = np.mean(reward_history[-100:])

        if len(agent_update_times) > 0:
            avg_agent_update_time = np.mean(agent_update_times[-100:]).total_seconds()
        else:
            avg_agent_update_time = 0

        if avg_reward > best_reward and len(reward_history) >= 100:
            best_reward = avg_reward
            
        
        if (total_episodes+1) % print_freq == 0:
            print("============================================================================================")
            print(f"Episode: {episode_num} \t Total Steps: {global_step_num} \t Average Reward: {avg_reward:9.02f} \t Best Reward: {best_reward:.02f}") 
            print(f"Elapsed Time: {datetime.now().replace(microsecond=0) - start_time} \t Avg Episode Time: {avg_episode_time:.2f}s \t Avg Agent Update Time: {avg_agent_update_time:.2f}s")
            print("============================================================================================")

        writer.add_scalar('Reward', episode_reward, global_step=global_step_num)
        writer.add_scalar('Average Reward', avg_reward, global_step=global_step_num)
        writer.add_scalar('Agent Train Time(s)', avg_agent_update_time, global_step=global_step_num)

        total_episodes += 1

        episode_num += 1

writer.close()
env.close()

end_time = datetime.now().replace(microsecond=0)
print()
print("Started training at: ", start_time)
print("Finished training at: ", end_time)
print("Total training time: ", end_time - start_time)