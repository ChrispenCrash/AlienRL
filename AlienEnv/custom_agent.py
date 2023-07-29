import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class AlienRL(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(AlienRL, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        self.fcn1 = nn.Sequential(
            nn.Linear(20, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(128 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),
        )

    def forward(self, obs):
        image = obs['image']
        telemetry = obs['telemetry']

        image_embedding = self.cnn(image)
        telemetry_embedding = self.fcn1(telemetry)

        if len(telemetry_embedding.shape) == 1:
            telemetry_embedding = telemetry_embedding.unsqueeze(0)

        concatenated = torch.cat((image_embedding, telemetry_embedding), dim=1)

        return self.fc2(concatenated)


if __name__ == "__main__":

    def count_params(model):
        return print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    observation_space = spaces.Dict({
            'framestack': spaces.Box(low=0, high=1, shape=(4,3,84,84), dtype=np.float32),
            'telemetry': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        })

    model = AlienRL(observation_space, 2)
    count_params(model)

    test_framestack = torch.randn(1, 4, 3, 84, 84)  # add a batch dimension
    test_framestack = test_framestack.view(1, -1, 84, 84)
    print(test_framestack.shape)

    test_telemetry = torch.randn(1, 20)  # add a batch dimension
    print(test_telemetry.shape)

    obs = {'image': test_framestack, 'telemetry': test_telemetry}

    print(model.cnn(test_framestack).shape)

    print(model.fcn1(test_telemetry).shape)

    # Switch to eval mode
    model.eval()
    print(model(obs))

    # Switch to train mode
    model.train()
    print(model(obs))