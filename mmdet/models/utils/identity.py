import torch


class IdentityModule(torch.nn.Module):
    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x):
        return x