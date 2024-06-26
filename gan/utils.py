import math

from torch import nn


class GeneratorMLP(nn.Module):
    def __init__(self, latent_dim, n_hidden, hidden_dim, out_shape):
        super().__init__()
        self.in_dim = latent_dim
        self.out_dim = math.prod(out_shape)
        self.n_channels, self.height, self.width = out_shape
        # add first hidden layer;
        self.net = nn.Sequential()
        self.net.add_module("hidden_0", nn.Linear(self.in_dim, hidden_dim))
        self.net.add_module("gelu_0", nn.GELU(approximate="tanh"))
        # add remaining hidden layers;
        for i in range(n_hidden - 1):
            self.net.add_module(
                f"hidden_{i+1}", nn.Linear(hidden_dim, hidden_dim)
            )
            self.net.add_module(f"gelu_{i+1}", nn.GELU(approximate="tanh"))
        self.net.add_module(f"out_layer", nn.Linear(hidden_dim, self.out_dim))

    def forward(self, x):
        return self.net(x).view(-1, self.n_channels, self.height, self.width)


class DiscriminatorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(1),  # 28 x 28
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(3),  # 12 x 12
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(5),  # 5 x 5
            nn.Flatten(),
            nn.Linear(5**3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
