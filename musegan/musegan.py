from musegan.generator import Generator
from musegan.critic import Critic
from musegan.loss import WassersteinLoss, GradientPenalty

import time
from tqdm import tqdm

import torch
from torch import nn


class MuseGAN(object):
    def __init__(self, z_dimension, g_channels,
                 g_features, c_channels, c_features,
                 n_bars, n_pitches, n_tracks, n_steps_per_bar,
                 g_lr=0.001, c_lr=0.001, device="cuda:0"):

        self.z_dimension = z_dimension
        self.n_tracks = n_tracks

        # generator and optimizer
        self.generator = Generator(z_dim=z_dimension, hid_chans=g_channels,
                                   hid_feats=g_features, out_chans=1,
                                   n_bars=n_bars, n_pitches=n_pitches,
                                   n_tracks=n_tracks,
                                   n_steps_per_bar=n_steps_per_bar).to(device)
        self.generator = self.generator.apply(self.initialize_weights)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(),
                                            lr=g_lr, betas=(0.5, 0.9))

        # critic and optimizer
        self.critic = Critic(hid_chans=c_channels, hid_feats=c_features,
                             out_feats=1, n_bars=n_bars,
                             n_pitches=n_pitches, n_tracks=n_tracks,
                             n_steps_per_bar=n_steps_per_bar).to(device)
        self.critic = self.critic.apply(self.initialize_weights)
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=c_lr, betas=(0.5, 0.9))

        # loss
        self.g_criterion = WassersteinLoss().to(device)
        self.c_criterion = WassersteinLoss().to(device)
        self.c_penalty = GradientPenalty().to(device)
        self.device = device

        # History
        self.history = {"g_loss": [],   # Generator loss
                        "c_loss": [],   # Critic loss
                        "cf_loss": [],  # Critic loss on fake images
                        "cr_loss": [],  # Critic loss on real images
                        "cp_loss": []}  # Critic gradient penalty
        print("MuseGAN initialized")

    @staticmethod
    def initialize_weights(layer, mean=0.0, std=0.02):
        if isinstance(layer, (nn.Conv3d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight, mean, std)
        elif isinstance(layer, (nn.Linear, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight, mean, std)
            nn.init.constant_(layer.bias, 0)

    def train(self, dataloader, epochs=500, batch_size=16, display_epoch=10):
        self.alpha = torch.randn((batch_size, 1, 1, 1, 1)).requires_grad_().to(self.device)

        trange = tqdm(range(epochs), position=0, leave=True)
        for epoch in trange:
            ge_loss, ce_loss = 0.0, 0.0
            cfe_loss, cre_loss, cpe_loss = 0, 0, 0
            start = time.time()
            for real in dataloader:
                real = real.to(self.device)

                # train Critic
                cb_loss = 0
                cfb_loss, crb_loss, cpb_loss = 0, 0, 0
                for _ in range(5):
                    cords = torch.randn(batch_size, 32).to(self.device)
                    style = torch.randn(batch_size, 32).to(self.device)
                    melody = torch.randn(batch_size, 4, 32).to(self.device)
                    groove = torch.randn(batch_size, 4, 32).to(self.device)
                    # forward to generator
                    self.c_optimizer.zero_grad()
                    with torch.no_grad():
                        fake = self.generator(cords, style, melody, groove).detach()
                    # mix `real` and `fake` melody
                    realfake = self.alpha * real + (1. - self.alpha) * fake
                    # get critic's `fake` loss
                    fake_pred = self.critic(fake)
                    fake_target = - torch.ones_like(fake_pred)
                    fake_loss = self.c_criterion(fake_pred, fake_target)
                    # get critic's `real` loss
                    real_pred = self.critic(real)
                    real_target = torch.ones_like(real_pred)
                    real_loss = self.c_criterion(real_pred, real_target)
                    # get critic's penalty
                    realfake_pred = self.critic(realfake)
                    penalty = self.c_penalty(realfake, realfake_pred)
                    # sum up losses
                    closs = fake_loss + real_loss + 10 * penalty
                    # retain graph
                    closs.backward(retain_graph=True)
                    # update critic parameters
                    self.c_optimizer.step()
                    # devide by number of critic updates in the loop (5)
                    cfb_loss += fake_loss.item() / 5
                    crb_loss += real_loss.item() / 5
                    cpb_loss += 10 * penalty.item() / 5
                    cb_loss += closs.item() / 5

                cfe_loss += cfb_loss / len(dataloader)
                cre_loss += crb_loss / len(dataloader)
                cpe_loss += cpb_loss / len(dataloader)
                ce_loss += cb_loss / len(dataloader)

                # train generator
                self.g_optimizer.zero_grad()
                # create random `noises`
                cords = torch.randn(batch_size, 32).to(self.device)
                style = torch.randn(batch_size, 32).to(self.device)
                melody = torch.randn(batch_size, 4, 32).to(self.device)
                groove = torch.randn(batch_size, 4, 32).to(self.device)
                # forward to generator
                fake = self.generator(cords, style, melody, groove)
                # forward to critic (to make prediction)
                fake_pred = self.critic(fake)
                # get generator loss (idea is to fool critic)
                gb_loss = self.g_criterion(fake_pred, torch.ones_like(fake_pred))
                gb_loss.backward()
                # update generator parameters
                self.g_optimizer.step()
                ge_loss += gb_loss.item() / len(dataloader)

            total_time = time.time() - start
            self.history['g_loss'].append(ge_loss)
            self.history['c_loss'].append(ce_loss)
            self.history['cf_loss'].append(cfe_loss)
            self.history['cr_loss'].append(cre_loss)
            self.history['cp_loss'].append(cpe_loss)

            if epoch % display_epoch == 0:
                message = f"Epoch {epoch}/{epochs} G loss: {ge_loss:.1f} D loss: {ce_loss:.1f} ETA: {total_time:.1f}s"
                message += " "
                message += f"[C loss | (fake: {cfe_loss:.1f}, real: {cre_loss:.1f}, penalty: {cpe_loss:.1f})]"
                trange.set_description(message, refresh=True)
                # print(f"Epoch {epoch}/{epochs} G loss: {ge_loss:.3f} D loss: {ce_loss:.3f} ETA: {total_time:.3f}s")
                # print(f"[C loss | (fake: {cfe_loss:.3f}, real: {cre_loss:.3f}, penalty: {cpe_loss:.3f})]")