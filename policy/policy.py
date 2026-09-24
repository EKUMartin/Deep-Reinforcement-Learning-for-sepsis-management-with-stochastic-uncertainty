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


class HighLevelQNetwork(nn.Module):
    def __init__(
        self,
        latent_dim=7,
        num_options=2
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            latent_dim + 1,
            64
        )

        self.fc2 = nn.Linear(
            64,
            64
        )

        self.out = nn.Linear(
            64,
            num_options
        )

        nn.init.orthogonal_(
            self.fc1.weight,
            gain=1.0
        )

        nn.init.zeros_(
            self.fc1.bias
        )

        nn.init.orthogonal_(
            self.fc2.weight,
            gain=1.0
        )

        nn.init.zeros_(
            self.fc2.bias
        )

        nn.init.orthogonal_(
            self.out.weight,
            gain=0.01
        )

        nn.init.zeros_(
            self.out.bias
        )

    def forward(
        self,
        z,
        uncertainty
    ):
        if uncertainty.dim() == 1:
            uncertainty = (
                uncertainty.unsqueeze(1)
            )

        x = torch.cat(
            [
                z,
                uncertainty
            ],
            dim=1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = F.relu(
            self.fc2(x)
        )

        return self.out(x)

    def select_option(
        self,
        z,
        uncertainty
    ):
        option_q = self.forward(
            z,
            uncertainty
        )

        option = option_q.argmax(
            dim=1
        )

        return (
            option,
            option_q
        )

    def select_action(
        self,
        z,
        uncertainty,
        q_net,
        p_net
    ):
        option_q = self.forward(
            z,
            uncertainty
        )

        option = option_q.argmax(
            dim=1
        )

        q_values = q_net(z)
        p_values = p_net(z)

        q_action = q_values.argmax(
            dim=1
        )

        p_action = p_values.argmax(
            dim=1
        )

        final_action = torch.where(
            option == 0,
            q_action,
            p_action
        )

        return (
            final_action,
            option,
            option_q
        )