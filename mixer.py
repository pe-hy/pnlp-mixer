import torch
import torch.nn as nn

class Mixer(nn.Module): 
    
    def __init__(self, num_mixers: int, max_seq_len: int, hidden_dim: int, mlp_hidden_dim: int, **kwargs):
        super(Mixer, self).__init__(**kwargs)
        self.mixers = nn.Sequential(*[
            MixerLayer(max_seq_len, hidden_dim, mlp_hidden_dim, mlp_hidden_dim) for _ in range(num_mixers)
        ])
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        return self.mixers(inputs)

class MixerLayer(nn.Module): 

    def __init__(self, max_seq_len: int, hidden_dim: int, channel_hidden_dim: int, seq_hidden_dim: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        # self.mlp_1 = FFFTrainFixed(max_seq_len, max_seq_len, 10)
        # self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        # self.mlp_2 = FFFTrainFixed(hidden_dim, hidden_dim, 10)
        self.mlp_1 = MlpLayer(max_seq_len, seq_hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp_2 = MlpLayer(hidden_dim, channel_hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        residual = inputs
        print(inputs.shape)
        a = inputs.shape[0]
        b = inputs.shape[2]
        c = inputs.shape[1]
        outputs = self.layer_norm_1(inputs)
        outputs = outputs.transpose(-1, -2)
        outputs = self.mlp_1(outputs)
        outputs = outputs.view(a, b, c)
        outputs = outputs.transpose(-1, -2) + residual
        residual = outputs
        outputs = self.layer_norm_2(outputs)
        outputs = self.mlp_2(outputs)
        outputs = outputs.view(inputs.shape)
        outputs = outputs + residual
        return outputs

class MlpLayer(nn.Module): 

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim), 
            nn.GELU(), 
            nn.Linear(intermediate_dim, hidden_dim)
        ])
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        return self.layers(inputs)


from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
import torch


class FFFTrainFixed(nn.Module):
    def __init__(self, input_width, output_width, depth, activation=nn.GELU):
        super().__init__()

        self.input_width = input_width
        self.output_width = output_width
        self.depth = depth  # this functions as the max depth
        self.parallel_size = 1  # parallel trees are not currently suppported.
        self.n_nodes = 2 ** (self.depth + 1) - 1
        self.bias = False
        self.switch_decisions_regularization = False
        if self.bias:  # easier to do bias like this in c++
            self.linear_in = nn.Linear(
                input_width + 1, self.parallel_size * self.n_nodes, bias=False
            )

        else:
            self.linear_in = nn.Linear(
                input_width, self.parallel_size * self.n_nodes, bias=False
            )
        self.linear_out = nn.Linear(
            self.parallel_size * self.n_nodes, output_width, bias=False
        )

        self.activation = activation()

    def forward(self, oldx: torch.Tensor) -> torch.Tensor:
        depth = self.depth

        x = oldx.reshape(-1, self.input_width)

        if self.bias:
            biastensor = torch.ones(x.shape[0], 1)
            x = torch.cat((x, biastensor), dim=1)

        batch_size = x.shape[0]

        logits = self.linear_in(x)  # (batch_size, parallel_size * n_nodes)

        logit_decisions = (logits > 0).long()  # (batch_size, parallel_size * n_nodes)

        if self.switch_decisions_regularization:
            flips = torch.tensor(
                np.random.choice([0, 1], logit_decisions.shape, p=[0.9, 0.1])
            )

            logit_decisions = logit_decisions + flips

            logit_decisions[logit_decisions == 2] = 0

        activations = self.activation(logits)  # (batch_size, parallel_size * n_nodes)

        activations = activations.view(
            batch_size, self.parallel_size, self.n_nodes
        )  # (batch_size, parallel_size, n_nodes)

        decisions = logit_decisions.view(
            batch_size, self.parallel_size, self.n_nodes
        )  # (batch_size, parallel_size, n_nodes)

        with torch.no_grad():
            current_nodes = torch.zeros(
                (batch_size, self.parallel_size), dtype=torch.long, device=x.device
            )

            decision_map = torch.zeros_like(
                decisions, dtype=torch.float
            )  # (batch_size, parallel_size, n_nodes)

            decision_map.scatter_(dim=2, index=current_nodes.unsqueeze(-1), value=1.0)

            for d in range(depth):
                current_platform = 2**d - 1

                next_platform = 2 ** (d + 1) - 1

                moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(
                    2
                )

                next_nodes = (
                    (current_nodes - current_platform) * 2 + moves + next_platform
                )

                decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)

                current_nodes = next_nodes

        activations = activations * decision_map  # (batch_size, parallel_size, n_nodes)

        new_logits = self.linear_out(
            activations.flatten(1, 2)
        )  # (batch_size, output_width)

        return new_logits
