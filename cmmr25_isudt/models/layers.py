import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, chans_per_group=16):
        super(ResBlock, self).__init__()

        # this is the residual block
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.GroupNorm(channel // chans_per_group, channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.GroupNorm(in_channel // chans_per_group, in_channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input  # skip connection
        return out


class ResBlock1D(nn.Module):
    def __init__(self, in_channel, channel, chans_per_group=16):
        super(ResBlock1D, self).__init__()

        # this is the residual block
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.GroupNorm(channel // chans_per_group, channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel, in_channel, 1),
            nn.GroupNorm(in_channel // chans_per_group, in_channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input  # skip connection
        return out
    

class LinearDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LinearDiscriminator, self).__init__()

        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            input_dim = hidden_dim
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        ])

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)


class LinearProjector(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_features=[64, 128, 256, 128, 64]):
        super(LinearProjector, self).__init__()

        layers = []
        features_per_group = 16
        num_groups = max(1, hidden_layers_features[0] // features_per_group)
        # add first layer
        layers.extend([
            nn.Linear(in_features, hidden_layers_features[0]),
            nn.GroupNorm(num_groups, hidden_layers_features[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # add hidden layers
        for i in range(len(hidden_layers_features)-1):
            num_groups = max(1, hidden_layers_features[i+1] // features_per_group)
            layers.extend([
                nn.Linear(hidden_layers_features[i],
                          hidden_layers_features[i+1]),
                nn.GroupNorm(num_groups, hidden_layers_features[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # add output layer
        layers.append(nn.Linear(hidden_layers_features[-1], out_features))

        self.linear_projector = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_projector(x)
    

class MultiScaleEncoder(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=128,
            n_res_block=1,
            n_res_channel=32,
            stride=4,
            kernels=[4, 4],
            input_dim_h=80,
            input_dim_w=188,
    ):
        super(MultiScaleEncoder, self).__init__()

        # check that the stride is valid
        assert stride in [2, 4]

        # check that kernels is a list with even number of elements
        assert len(kernels) % 2 == 0

        # group kernels into pairs
        kernels = [kernels[i:i + 2] for i in range(0, len(kernels), 2)]

        # save input dimension for later use
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w

        # create a list of lanes
        self.lanes = nn.ModuleList()

        # create a lane for each kernel size
        for kernel in kernels:
            padding = [max(1, (kernel_side - 1) // 2) for kernel_side in kernel]
            lane = None

            if stride == 4:
                # base block: in -> out/2 -> out -> out
                lane = [
                    nn.Conv2d(in_channel, channel // 2, kernel, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channel // 2, channel, kernel, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, stride=[1, 2], padding=1),
                ]

            elif stride == 2:
                # base block: in -> out/2 -> out
                lane = [
                    nn.Conv2d(in_channel, channel // 2, kernel,
                              stride=2, padding=padding),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channel // 2, channel, 3, padding=1),
                ]

            # add residual blocks
            lane.extend([ResBlock(channel, n_res_channel)
                         for _ in range(n_res_block)])

            # add final ReLU
            lane.append(nn.LeakyReLU(inplace=True))

            # add to list of blocks
            self.lanes.append(nn.Sequential(*lane))
    
    def forward(self, input):
        # Process all lanes and stack the results
        outputs = [lane(input) for lane in self.lanes]
        
        # Use torch's built-in sum function to add all outputs together
        if len(outputs) == 1:
            return outputs[0]
        else:
            return torch.stack(outputs).sum(dim=0)


# loosely based on MultiScaleEncoder
class ConvEncoder1DRes(nn.Module):
    def __init__(
            self, 
            in_channel=1, 
            channel=128, 
            n_res_block=1, 
            n_res_channel=32,
            stride=4,
            kernel=3,
            ):
        super().__init__()

        # check that the stride is valid
        assert stride in [2, 4]

        padding = max(1, (kernel - 1) // 2)

        if stride == 4:
            # base block: in -> out/2 -> out -> out
            layers = [
                nn.Conv1d(in_channel, channel // 2, kernel, stride=2, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(channel // 2, channel, kernel, stride=2, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(channel, channel, 3, stride=1, padding=1),
            ]

        elif stride == 2:
            # base block: in -> out/2 -> out
            layers = [
                nn.Conv1d(in_channel, channel // 2, kernel, stride=2, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(channel // 2, channel, 3, padding=1),
            ]

        # add residual blocks
        layers.extend([ResBlock1D(channel, n_res_channel) for _ in range(n_res_block)])

        # add final ReLU
        layers.append(nn.LeakyReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)