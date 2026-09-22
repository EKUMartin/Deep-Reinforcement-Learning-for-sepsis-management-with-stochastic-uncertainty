import torch
import torch.nn as nn
import torch.nn.functional as F


class LowLevelQNetwork(nn.Module):
    def __init__(self, latent_dim=7, action_dim=25):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)

        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        return self.out(x)


class HighLevelPolicy(nn.Module):
    def __init__(self, latent_dim=7, num_policies=2):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_policies)

        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)

        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, g_norm):
        if g_norm.dim() == 1:
            g_norm = g_norm.unsqueeze(1)

        x = torch.cat([z, g_norm], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.out(x)

        return F.softmax(logits, dim=1)

    def compute_loss(
        self,
        probs,
        q_values,
        p_values,
        g_norm,
        alpha=0.25,
        beta=0.01
    ):
        q_value = q_values.max(dim=1).values.detach()
        p_value = p_values.max(dim=1).values.detach()

        option_values = torch.stack(
            [q_value, p_value],
            dim=1
        )

        mean = option_values.mean()
        std = option_values.std(unbiased=False) + 1e-6

        option_values = (
            option_values - mean
        ) / std

        expected_value = torch.sum(
            probs * option_values,
            dim=1
        )

        policy_loss = -expected_value.mean()

        prob_prospect = probs[:, 1]

        guidance_loss = F.binary_cross_entropy(
            prob_prospect,
            g_norm.detach()
        )

        entropy = -torch.sum(
            probs * torch.log(probs + 1e-8),
            dim=1
        ).mean()

        total_loss = (
            policy_loss
            + alpha * guidance_loss
            - beta * entropy
        )

        return (
            total_loss,
            policy_loss,
            guidance_loss,
            entropy
        )

    def select_policy(self, z, g_norm):
        probs = self.forward(z, g_norm)
        policy = probs.argmax(dim=1)

        return policy, probs

    def select_action(
        self,
        z,
        g_norm,
        q_net,
        p_net
    ):
        probs = self.forward(
            z,
            g_norm
        )

        policy = probs.argmax(dim=1)

        q_values = q_net(z)
        p_values = p_net(z)

        q_action = q_values.argmax(dim=1)
        p_action = p_values.argmax(dim=1)

        final_action = torch.where(
            policy == 0,
            q_action,
            p_action
        )

        return (
            final_action,
            policy,
            probs
        )