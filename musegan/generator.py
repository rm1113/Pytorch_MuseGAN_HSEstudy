import torch
from torch import nn



class Reshape(nn.Module):
    """
    Reshape Layer for using inside the torch.nn.Sequential
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Conv2dNormReluLayer(nn.Module):
    """
    The combination of:
        torch.nn.ConvTranspose2d
        torch.nn.BatchNorm2d
        torch.nn.ReLU
    for using inside the torch.nn.Sequential
    """
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim,
                               kernel_size=kernel,
                               stride=stride),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.network(x)
        return y


class TemporalNetwork(nn.Module):
    """
    Temporal submodel for MuseGAN
    """
    def __init__(self, z_dim=32, hid_chans=1024, n_bars=2):
        super().__init__()
        self.network = nn.Sequential(
            Reshape(shape=[z_dim, 1, 1]),

            Conv2dNormReluLayer(in_dim=z_dim, out_dim=hid_chans,
                                kernel=(2, 1), stride=(1, 1)),

            Conv2dNormReluLayer(in_dim=hid_chans, out_dim=z_dim,
                                kernel=(n_bars - 1, 1),
                                stride=(1, 1)),

            Reshape(shape=[z_dim, n_bars]))

    def forward(self, x):
        y = self.network(x)
        return y


class BarGenerator(nn.Module):
    """
    Bar generator submodel for MuseGAN
    """

    def __init__(self, z_dim=32, hid_chans=512, hid_feats=1024,
                 out_chans=1, n_step_per_bar=16, n_pitches=84):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4 * z_dim, hid_feats),
            nn.BatchNorm1d(hid_feats),
            nn.ReLU(inplace=True),

            Reshape(shape=[hid_chans, hid_feats // hid_chans, 1]),
            Conv2dNormReluLayer(in_dim=hid_chans, out_dim=hid_chans,
                                kernel=(2, 1), stride=(2, 1)),

            Conv2dNormReluLayer(in_dim=hid_chans, out_dim=hid_chans // 2,
                                kernel=(2, 1), stride=(2, 1)),

            Conv2dNormReluLayer(in_dim=hid_chans // 2, out_dim=hid_chans // 2,
                                kernel=(2, 1), stride=(2, 1)),

            Conv2dNormReluLayer(in_dim=hid_chans // 2, out_dim=hid_chans // 2,
                                kernel=(1, 7), stride=(1, 7)),

            nn.ConvTranspose2d(hid_chans // 2, out_chans,
                               kernel_size=(1, 12), stride=(1, 12)),
            Reshape(shape=[1, 1, n_step_per_bar, n_pitches])
        )

    def forward(self, x):
        y = self.network(x)
        return y


class Generator(torch.nn.Module):
    """
    The MuseGAN generator.
    """
    def __init__(self, z_dim=32, hid_chans=1024, hid_feats=1024,
                 out_chans=1, n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84):
        super().__init__()

        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        self.chords_network = TemporalNetwork(z_dim=z_dim,
                                              hid_chans=hid_chans,
                                              n_bars=n_bars)
        # melody generator
        self.melody_networks = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.melody_networks.add_module(
                f"melodygen_{n}",
                TemporalNetwork(z_dim, hid_chans, n_bars),
            )

        # bar generator
        self.bar_generators = nn.ModuleDict({})
        for n in range(self.n_tracks):
            self.bar_generators.add_module(
                f"bargen_{n}",
                BarGenerator(z_dim=z_dim,
                             hid_chans=hid_chans // 2,
                             hid_feats=hid_feats,
                             out_chans=out_chans,
                             n_step_per_bar=n_steps_per_bar,
                             n_pitches=n_pitches
                             )
            )

    def forward(self, chords, style, melody, groove):
        # chords shape: (batch_size, z_dimension)
        # style shape: (batch_size, z_dimension)
        # melody shape: (batch_size, n_tracks, z_dimension)
        # groove shape: (batch_size, n_tracks, z_dimension)
        chord_outs = self.chords_network(chords)
        bar_outs = []
        for bar in range(self.n_bars):
            track_outs = []
            chord_out = chord_outs[:, :, bar]
            style_out = style
            for track in range(self.n_tracks):
                melody_in = melody[:, track, :]
                melody_out = self.melody_networks[f"melodygen_{track}"](melody_in)[:, :, bar]
                groove_out = groove[:, track, :]

                z = torch.cat([chord_out, style_out, melody_out, groove_out], dim=1)
                track_outs.append(self.bar_generators[f"bargen_{track}"](z))
            track_out = torch.cat(track_outs, dim=1)
            bar_outs.append(track_out)
        out = torch.cat(bar_outs, dim=2)
        return out
