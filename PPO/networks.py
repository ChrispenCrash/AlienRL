import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# class ActorNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorNetwork, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, action_dim),
#             nn.Tanh()
#         )
        
#     def forward(self, state):
#         return self.network(state)

# class CriticNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super(CriticNetwork, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
        
#     def forward(self, state):
#         return self.network(state)


# Second attempt
# class ActorNetwork(nn.Module):
#     def __init__(self):
#         super(ActorNetwork, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(12, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3136, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 128),
#             nn.ReLU()
#         )
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(20, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(128, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(p=0.2)
#         )
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(128 * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2),
#             nn.Tanh(),
#         )

#     def forward(self, obs):
#         framestack = obs['framestack']
#         telemetry = obs['telemetry']
        
#         framestack_embedding = self.cnn(framestack)
#         telemetry_embedding = self.fc1(telemetry)

#         if len(telemetry_embedding.shape) == 1:
#             telemetry_embedding = telemetry_embedding.unsqueeze(0)

#         concatenated = torch.cat((framestack_embedding, telemetry_embedding), dim=1)

#         return self.fc2(concatenated)
    
# class CriticNetwork(nn.Module):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(12, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3136, 1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 128),
#             nn.ReLU()
#         )
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(20, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(128, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(p=0.2)
#         )
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(128 * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, obs):
#         framestack = obs['framestack']
#         telemetry = obs['telemetry']

#         framestack_embedding = self.cnn(framestack)
#         telemetry_embedding = self.fc1(telemetry)

#         if len(telemetry_embedding.shape) == 1:
#             telemetry_embedding = telemetry_embedding.unsqueeze(0)

#         concatenated = torch.cat((framestack_embedding, telemetry_embedding), dim=1)

#         return self.fc2(concatenated)
    

# Third attempt
# class ActorNetwork(nn.Module):
#     def __init__(self):
#         super(ActorNetwork, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(12, 16, kernel_size=8, stride=4),
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1),
#             nn.LeakyReLU(),
#             nn.Flatten(),
#             nn.Linear(1568, 512),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 64),
#             nn.LeakyReLU(),
#         )
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(64, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2)
#         )
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(64 * 2, 256),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(128, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 2),
#             nn.Tanh(),
#         )

#     def forward(self, obs):
#         framestack = obs['framestack']
#         telemetry = obs['telemetry']
        
#         framestack_embedding = self.cnn(framestack)
#         telemetry_embedding = self.fc1(telemetry)

#         if len(telemetry_embedding.shape) == 1:
#             telemetry_embedding = telemetry_embedding.unsqueeze(0)

#         concatenated = torch.cat((framestack_embedding, telemetry_embedding), dim=1)

#         return self.fc2(concatenated)
    
# class CriticNetwork(nn.Module):
#     def __init__(self):
#         super(CriticNetwork, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(12, 16, kernel_size=8, stride=4),
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1),
#             nn.LeakyReLU(),
#             nn.Flatten(),
#             nn.Linear(1568, 512),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 64),
#             nn.LeakyReLU(),
#         )
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(64, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2)
#         )
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(64 * 2, 256),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(128, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 1)
#         )

#     def forward(self, obs):
#         framestack = obs['framestack']
#         telemetry = obs['telemetry']

#         framestack_embedding = self.cnn(framestack)
#         telemetry_embedding = self.fc1(telemetry)

#         if len(telemetry_embedding.shape) == 1:
#             telemetry_embedding = telemetry_embedding.unsqueeze(0)

#         concatenated = torch.cat((framestack_embedding, telemetry_embedding), dim=1)

#         return self.fc2(concatenated)
    

# # Third attempt
# class ActorNetwork(nn.Module):
#     def __init__(self):
#         super(ActorNetwork, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(12, 16, kernel_size=8, stride=4),
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 32, kernel_size=4, stride=2),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1),
#             nn.LeakyReLU(),
#             nn.Flatten(),
#             nn.Linear(1568, 512),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(512, 64),
#             nn.LeakyReLU(),
#         )
        
#         self.fc1 = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(64, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2)
#         )
        
#         self.fc2 = nn.Sequential(
#             nn.Linear(64 * 2, 256),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(128, 32),
#             nn.LeakyReLU(),
#             nn.Linear(32, 2),
#             nn.Tanh(),
#         )

#     def forward(self, obs):
#         framestack = obs['framestack']
#         telemetry = obs['telemetry']
        
#         framestack_embedding = self.cnn(framestack)
#         telemetry_embedding = self.fc1(telemetry)

#         if len(telemetry_embedding.shape) == 1:
#             telemetry_embedding = telemetry_embedding.unsqueeze(0)

#         concatenated = torch.cat((framestack_embedding, telemetry_embedding), dim=1)

#         return self.fc2(concatenated)
    

# Fourth attempt
class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        
        # Convolutional layers for image
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1568, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )
        
        # Fully connected layers for telemetry data
        self.fc1 = nn.Sequential(
            nn.Linear(9, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        
        self.lstm = nn.LSTM(256, hidden_size=128, num_layers=2, batch_first=True)
        
        # Fully connected layer for output
        self.fc_out = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        # Initialize weights
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    
    def forward(self, obs):

        image = obs['image']
        telemetry = obs['telemetry']

        cnn = self.cnn(image)
        fc1 = self.fc1(telemetry)
        
        combined = torch.cat((cnn, fc1), dim=1)
        
        lstm_out, _ = self.lstm(combined)
        
        output = self.fc_out(lstm_out.squeeze(1))
        
        return output
    

class CriticNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        
        # Convolutional layers for image
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1568, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU()
        )
        
        # Fully connected layers for telemetry data
        self.fc1 = nn.Sequential(
            nn.Linear(9, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU()
        )
        
        self.lstm = nn.LSTM(256, hidden_size=128, num_layers=2, batch_first=True)
        
        # Fully connected layer for output
        self.fc_out = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

        # Initialize weights
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

    
    def forward(self, obs):

        image = obs['image']
        telemetry = obs['telemetry']

        cnn = self.cnn(image)
        fc1 = self.fc1(telemetry)
        
        combined = torch.cat((cnn, fc1), dim=1)
        
        lstm_out, _ = self.lstm(combined)
        
        output = self.fc_out(lstm_out.squeeze(1))
        
        return output