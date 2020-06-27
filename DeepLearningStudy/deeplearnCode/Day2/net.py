import torch
from torch import nn


class NetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU()
        )

        self.liner = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        h = self.seq(x)
        h = h.reshape(-1, 128 * 4 * 4)

        return self.liner(h)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)




class NetV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU()
        )

        self.liner = nn.Linear(128 * 4 * 4, 10)

        self.apply(weight_init)

    def forward(self, x):
        h = self.seq(x)
        h = h.reshape(-1, 128 * 4 * 4)

        return self.liner(h)


if __name__ == '__main__':
    net = NetV2()
    # x = torch.rand(1, 3, 32, 32)
    # y = net(x)
    # print(y.shape)
