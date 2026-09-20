import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
# ==========================================
# Low-Level Network (Q-learning & Prospect)
# ==========================================
class LowLevelQNetwork(nn.Module):
    def __init__(self, latent_dim=8, action_dim=25):
        super(LowLevelQNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values


# ==========================================
#  High-Level Network
# ==========================================
class HighLevelPolicy(nn.Module):
    def __init__(self, latent_dim=8, num_policies=2):
        super(HighLevelPolicy, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_policies)

    def forward(self, z, g_norm):
        if g_norm.dim() == 1:
            g_norm = g_norm.unsqueeze(1)
            
        x = torch.cat([z, g_norm], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        probs = F.softmax(logits, dim=1)
        return probs

    def get_action(self, z, g_norm):
        probs = self.forward(z, g_norm)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action), probs

    def compute_loss(self, log_prob, probs, reward, g_norm, baseline_reward=0, alpha=0.5, beta=0.05):
        advantage = reward - baseline_reward
        policy_loss = -log_prob * advantage 
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        entropy_loss = -beta * entropy
        prob_prospect = probs[:, 1]
        guidance_loss = alpha * F.mse_loss(prob_prospect, g_norm.squeeze())
        total_loss = policy_loss.mean() + entropy_loss.mean() + guidance_loss      
        return total_loss, policy_loss.mean(), entropy_loss.mean(), guidance_loss