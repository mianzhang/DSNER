import torch.nn as nn


class Selector(nn.Module):

    def __init__(self, args):
        super(Selector, self).__init__()
        input_dim = args.hidden_dim * args.max_len + args.max_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, state_rep):
        out = self.mlp(state_rep)
        return out
