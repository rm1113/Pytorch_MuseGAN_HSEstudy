from torch import nn


class Critic(nn.Module):
    """
    The critic for MuseGAN
    """
    def __init__(self, hid_chans=128, hid_feats=1024, out_feats=1,
                 n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84):
        super().__init__()

        in_features = 4 * hid_chans if n_bars == 2 else 12 * hid_chans
        self.network = nn.Sequential(
            nn.Conv3d(n_tracks, hid_chans,
                      kernel_size=(2, 1, 1), stride=(1, 1, 1)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans, hid_chans,
                      kernel_size=(n_bars - 1, 1, 1), stride=(1, 1, 1)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans, hid_chans,
                      kernel_size=(1, 1, 12), stride=(1, 1, 12)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans, hid_chans,
                      kernel_size=(1, 1, 7), stride=(1, 1, 7)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans, hid_chans,
                      kernel_size=(1, 2, 1), stride=(1, 2, 1)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans, hid_chans,
                      kernel_size=(1, 2, 1), stride=(1, 2, 1)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans, hid_chans * 2,
                      kernel_size=(1, 4, 1), stride=(1, 2, 1),
                      padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Conv3d(hid_chans * 2, hid_chans * 4,
                      kernel_size=(1, 3, 1), stride=(1, 2, 1),
                      padding=(0, 1, 0)),
            nn.LeakyReLU(0.3, inplace=True),

            nn.Flatten(),
            nn.Linear(in_features, hid_feats),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(hid_feats, out_feats)
        )

    def forward(self, x):
        y = self.network(x)
        return y