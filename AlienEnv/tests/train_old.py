import keyboard
from custom_agent import AlienRL
import numpy as np
from gymnasium.spaces import Box
from gamestate import get_initial_observation


def main():
    # Hyperparameters 
    batch_size = 32
    alpha = 0.0001
    n_epochs = 1
    action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    observation_space = Box(low=0, high=255, shape=(720, 1280, 4), dtype=np.uint8)

    # Initialize the game state and agent
    agent = AlienRL(
        n_actions=action_space.n,
        input_dims=observation_space,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
    )

    

    (frame_obs, telemetry_obs) = get_initial_observation()

    observation = (frame_obs, telemetry_obs)

    while True:
        # Get the current game state
        action, prob, val = agent.choose_action(observation)
        observation, reward = agent.step(action)

        if keyboard.is_pressed("q"):
            print("Stopped.")
            break


if __name__ == "__main__":
    main()
