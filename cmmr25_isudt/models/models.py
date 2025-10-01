import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .layers import ResBlock, ResBlock1D, ConvEncoder1DRes, LinearDiscriminator, LinearProjector, MultiScaleEncoder
from .ddsp import Sinewave
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from lightning.pytorch import LightningModule
from ..utils.tensor import permute_dims, midi2frequency, scale
from .loss import kld_loss


class MelSpecEncoder(nn.Module):
    def __init__(
            self,
            input_width=512,
            encoder_channels=128,
            encoder_kernels=[4, 4],
            n_res_block=2,
            n_res_channel=32,
            stride=4,
            latent_size=8,
            dropout=0.2,
            ):
        super().__init__()

        self.chans_per_group = 16

        self.encoder = MultiScaleEncoder(
            in_channel=1,
            channel=encoder_channels,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride,
            kernels=encoder_kernels,
            input_dim_h=0,
            input_dim_w=0,
        )

        encoded_width = input_width // stride # 128
        target_width = 2
        num_blocks = int(np.log2(encoded_width)) - int(np.log2(target_width)) # 7 - 1 = 6

        post_encoder_blocks = []
        post_encoder_block = [
            nn.Conv2d(encoder_channels, encoder_channels, 3, 2, 1),
            nn.GroupNorm(encoder_channels // self.chans_per_group, encoder_channels),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_blocks):
            post_encoder_blocks.extend(post_encoder_block)

        self.post_encoder = nn.Sequential(*post_encoder_blocks)

        post_encoder_n_features = encoder_channels * target_width # 256
        target_n_features = latent_size * 2 # 16
        mlp_layers = []
        num_mlp_blocks = int(np.log2(post_encoder_n_features) - np.log2(target_n_features)) # 256 -> 16 = 4 blocks
        mlp_layers_features = [post_encoder_n_features // (2 ** i) for i in range(num_mlp_blocks + 1)]
        for i in range(num_mlp_blocks):
            num_groups = max(1, mlp_layers_features[i + 1] // self.chans_per_group)
            block = [
                nn.Linear(mlp_layers_features[i], mlp_layers_features[i + 1]),
                nn.GroupNorm(num_groups, mlp_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            mlp_layers.extend(block)

        self.mlp = nn.Sequential(*mlp_layers)

        self.mu = nn.Linear(target_n_features, latent_size)
        self.logvar = nn.Linear(target_n_features, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.post_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class MelSpec1DEncoder(nn.Module):
    def __init__(
            self,
            input_width=64,
            encoder_channels=128,
            encoder_kernels=3,
            n_res_block=2,
            n_res_channel=32,
            stride=4,
            latent_size=8,
            ):
        super().__init__()

        self.chans_per_group = 16

        self.encoder = ConvEncoder1DRes(
            in_channel=1,
            channel=encoder_channels,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            stride=stride,
            kernel=encoder_kernels,
        )

        encoded_width = input_width // stride # 16
        target_width = 2
        num_blocks = int(np.log2(encoded_width)) - int(np.log2(target_width)) # 4 - 1 = 3

        post_encoder_blocks = []
        post_encoder_block = [
            nn.Conv1d(encoder_channels, encoder_channels, 3, 2, 1),
            nn.GroupNorm(encoder_channels // self.chans_per_group, encoder_channels),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_blocks):
            post_encoder_blocks.extend(post_encoder_block)

        self.post_encoder = nn.Sequential(*post_encoder_blocks)

        post_encoder_n_features = encoder_channels * target_width # 256
        target_n_features = latent_size * 2 # 16
        mlp_layers = []
        num_mlp_blocks = int(np.log2(post_encoder_n_features) - np.log2(target_n_features))
        mlp_layers_features = [post_encoder_n_features // (2 ** i) for i in range(num_mlp_blocks + 1)]
        for i in range(num_mlp_blocks):
            num_groups = max(1, mlp_layers_features[i + 1] // self.chans_per_group)
            block = [
                nn.Linear(mlp_layers_features[i], mlp_layers_features[i + 1]),
                nn.GroupNorm(num_groups, mlp_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            mlp_layers.extend(block)
        
        self.mlp = nn.Sequential(*mlp_layers)

        self.mu = nn.Linear(target_n_features, latent_size)
        self.logvar = nn.Linear(target_n_features, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.post_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar


class MelSpec1DDecoder(nn.Module):
    def __init__(
            self,
            output_width=64,
            output_channels=1,
            decoder_channels=128,
            n_res_block=2,
            n_res_channel=32,
            latent_size=2,
            ):
        super().__init__()

        self.chans_per_group = 16 if decoder_channels >= 32 else decoder_channels // 2

        # convtranspose to output_width and add channels, do resblocks, predict
        num_convtranspose_blocks = int(np.log2(output_width) - np.log2(latent_size))  # 64, 8 => 6 - 1 = 5 blocks
        convtranspose_blocks = [nn.Unflatten(1, (1, latent_size))]  # (B, 1, 2)
        for i in range(num_convtranspose_blocks):
            in_channels = min(2 ** i, decoder_channels)
            out_channels = min(2 ** (i + 1), decoder_channels)
            num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
            convtranspose_blocks.extend([
                nn.ConvTranspose1d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2),
            ])
        upscaled_channels = min(2 ** num_convtranspose_blocks, decoder_channels)  # 64
        num_conv_blocks = int(np.log2(decoder_channels) - np.log2(upscaled_channels))  # 128, 64 => 7 - 6 = 1 block
        for i in range(num_conv_blocks):
            in_channels = upscaled_channels * (2 ** i)
            out_channels = upscaled_channels * (2 ** (i + 1))
            num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
            convtranspose_blocks.extend([
                nn.Conv1d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2),
            ])
        # remove the last leakyrelu
        convtranspose_blocks = convtranspose_blocks[:-1]
        self.upscaler = nn.Sequential(*convtranspose_blocks)

        resblocks = [ResBlock1D(decoder_channels, n_res_channel) for _ in range(n_res_block)]
        resblocks.append(
            nn.LeakyReLU(0.2)
        )
        self.res_blocks = nn.Sequential(*resblocks)

        num_head_blocks = int(np.log2(decoder_channels) - np.log2(output_channels))
        head_blocks = []
        for i in range(num_head_blocks - 1):
            in_channels = decoder_channels // (2 ** i)
            out_channels = decoder_channels // (2 ** (i + 1))
            num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
            head_blocks.extend([
                nn.Conv1d(in_channels, out_channels, 3, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2),
            ])
        head_blocks.extend([
            nn.Conv1d(decoder_channels // (2 ** (num_head_blocks - 1)), output_channels, 3, padding=1),
            nn.Identity()
        ])
        self.head = nn.Sequential(*head_blocks)

    def forward(self, z):
        x = self.upscaler(z)
        x = self.res_blocks(x)
        x = self.head(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(
            self,
            output_width=512,
            output_channels=1,
            decoder_channels=128,
            n_res_block=2,
            n_res_channel=64,
            latent_size=16,
            ):
        super().__init__()

        self.chans_per_group = 16 if decoder_channels >= 32 else decoder_channels // 2

        # mlp it up to 64, then reshape to 8x8, convtranspose to output_width, do resblocks, predict
        target_n_features = 64
        num_mlp_blocks = int(np.log2(target_n_features) - np.log2(latent_size))  # 64 -> 16 = 3 blocks
        mlp_layers = []
        mlp_layers_features = [latent_size * (2 ** i) for i in range(num_mlp_blocks + 1)]
        for i in range(num_mlp_blocks):
            num_groups = max(1, mlp_layers_features[i + 1] // self.chans_per_group)
            block = [
                nn.Linear(mlp_layers_features[i], mlp_layers_features[i + 1]),
                nn.GroupNorm(num_groups, mlp_layers_features[i + 1]),
                nn.LeakyReLU(0.2),
            ]
            mlp_layers.extend(block)

        self.mlp = nn.Sequential(*mlp_layers)

        reshaped_width = int(np.sqrt(target_n_features))  # 8
        reshape_layers = [nn.Unflatten(1, (1, reshaped_width, reshaped_width))]
        reshape_num_blocks = int(np.log2(decoder_channels))
        for i in range(reshape_num_blocks):
            in_channels = 2 ** i
            out_channels = 2 ** (i + 1)
            num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
            num_groups = max(1, num_groups)
            reshape_layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(num_groups, out_channels),
                nn.LeakyReLU(0.2),
            ])
        self.reshape = nn.Sequential(*reshape_layers)
        # now we have a 8x8 feature map with decoder_channels channels

        reshaped_width = int(np.sqrt(target_n_features))  # 8
        num_convtranspose_blocks = int(np.log2(output_width) - np.log2(reshaped_width))  # 512 -> 8 = 6 blocks
        convtranspose_blocks = []
        num_groups = max(1, decoder_channels // self.chans_per_group)
        convtranspose_block = [
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups, decoder_channels),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(num_convtranspose_blocks - 1):
            convtranspose_blocks.extend(convtranspose_block)
        # last block has no activation
        convtranspose_blocks.extend([
            nn.ConvTranspose2d(decoder_channels, decoder_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups, decoder_channels),
        ])
        self.upscaler = nn.Sequential(*convtranspose_blocks)
        # now we have a 512x512 feature map with decoder_channels channels

        resblocks = [ResBlock(decoder_channels, n_res_channel, self.chans_per_group) for _ in range(n_res_block)]
        resblocks.append(
            nn.LeakyReLU(0.2)  # last resblock has no activation
        )
        self.res_blocks = nn.Sequential(*resblocks)

        # round up output_channels to next power of 2
        output_channels_p2 = 2 ** int(max(1, np.ceil(np.log2(output_channels))))
        num_head_blocks = int(np.log2(decoder_channels) - np.log2(output_channels_p2))
        head_layers = []
        for i in range(num_head_blocks):
            in_channels = 2 ** int(np.log2(decoder_channels) - i)
            out_channels = 2 ** int(np.log2(decoder_channels) - i - 1)
            if i < num_head_blocks - 1:
                num_groups = out_channels // self.chans_per_group if out_channels // self.chans_per_group >= 2 else out_channels // 2
                num_groups = max(1, num_groups)
                head_layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.GroupNorm(num_groups, out_channels),
                    nn.LeakyReLU(0.2)
                ])
            else:
                out_channels = output_channels
                head_layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.Identity()  # output is unbounded
                ])

        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.mlp(x)
        x = self.reshape(x)
        x = self.upscaler(x)
        x = self.res_blocks(x)
        x = self.head(x)
        return x


class ImageVAE(nn.Module):
    def __init__(self,
                 input_width=512,
                 output_channels=1,
                 encoder_channels=128,
                 encoder_kernels=[4, 4],
                 encoder_n_res_block=2,
                 encoder_n_res_channel=64,
                 decoder_channels=128,
                 decoder_n_res_block=2,
                 decoder_n_res_channel=64,
                 latent_size=16,
                 ):
        super().__init__()
        self.encoder = MelSpecEncoder(
            input_width=input_width,
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            n_res_block=encoder_n_res_block,
            n_res_channel=encoder_n_res_channel,
            stride=4,
            latent_size=latent_size,
        )
        self.decoder = ImageDecoder(
            output_width=input_width,
            output_channels=output_channels,
            decoder_channels=decoder_channels,
            n_res_block=decoder_n_res_block,
            n_res_channel=decoder_n_res_channel,
            latent_size=latent_size
        )
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_img = self.decoder(z)
        return decoded_img, mu, logvar, z


class SineVAE(nn.Module):
    def __init__(self,
                 input_width=64,
                 encoder_channels=128,
                 encoder_kernels=3,
                 encoder_n_res_block=2,
                 encoder_n_res_channel=32,
                 decoder_channels=128,
                 decoder_n_res_block=2,
                 decoder_n_res_channel=32,
                 latent_size=8,
                 ):
        super().__init__()
        self.encoder = MelSpec1DEncoder(
            input_width=input_width,
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            n_res_block=encoder_n_res_block,
            n_res_channel=encoder_n_res_channel,
            stride=4,
            latent_size=latent_size,
        )
        self.decoder = MelSpec1DDecoder(
            output_width=input_width,
            output_channels=1,
            decoder_channels=decoder_channels,
            n_res_block=decoder_n_res_block,
            n_res_channel=decoder_n_res_channel,
            latent_size=latent_size
        )
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_x = self.decoder(z)
        return decoded_x, mu, logvar, z


class PlImgFactorVAE(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.args = args

        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.recon_loss_weight_start = args.recon_loss_weight_start
        self.recon_loss_weight = args.recon_loss_weight_start # initialize to start
        self.recon_loss_weight_end = args.recon_loss_weight_end
        self.recon_loss_weight_ramp_start_epoch = args.recon_loss_weight_ramp_start_epoch
        self.recon_loss_weight_ramp_end_epoch = args.recon_loss_weight_ramp_end_epoch
        self.latent_size = args.latent_size
        self.logdir = args.logdir
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.recon_loss = nn.MSELoss() if self.args.recon_loss_type == 'mse' else nn.L1Loss()
        self.recon_loss.eval()
        self.kld = kld_loss
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.tc_weight_max = args.tc_weight_max
        self.tc_weight_min = args.tc_weight_min
        self.tc_start_epoch = args.tc_start_epoch
        self.tc_warmup_epochs = args.tc_warmup_epochs

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # models
        self.model = ImageVAE(
            input_width=args.img_size,
            output_channels=1,
            encoder_channels=args.encoder_channels,
            encoder_kernels=args.encoder_kernels,
            encoder_n_res_block=args.encoder_n_res_block,
            encoder_n_res_channel=args.encoder_n_res_channel,
            decoder_channels=args.decoder_channels,
            decoder_n_res_block=args.decoder_n_res_block,
            decoder_n_res_channel=args.decoder_n_res_channel,
            latent_size=self.latent_size
        )
        self.D = LinearDiscriminator(self.latent_size, self.d_hidden_size, 2, self.d_num_layers)


        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)
        self.D.apply(init_weights_kaiming)
    

    def forward(self, x):
        # predict the image
        predicted_img, mu, logvar, z = self.model(x)
        return predicted_img, mu, logvar, z
    

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.D.train()

        # get the optimizers and schedulers
        vae_optimizer, d_optimizer = self.optimizers()
        vae_scheduler, d_scheduler = self.lr_schedulers()

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # get the batch
        epoch_idx = self.trainer.current_epoch
        x1, x2 = batch

        # forward pass
        # predict the image
        predicted_img, mu, logvar, z = self.model(x1)

        # VAE recon_loss
        recon_loss = self.recon_loss(predicted_img, x1)
        # calculate current recon loss weight
        current_epoch = self.trainer.current_epoch
        if current_epoch < self.recon_loss_weight_ramp_start_epoch:
            recon_loss_weight = self.recon_loss_weight_start
        elif current_epoch > self.recon_loss_weight_ramp_end_epoch:
            recon_loss_weight = self.recon_loss_weight_end
        else:
            recon_loss_weight = self.recon_loss_weight_start + (self.recon_loss_weight_end - self.recon_loss_weight_start) * \
                (current_epoch - self.recon_loss_weight_ramp_start_epoch) / \
                (self.recon_loss_weight_ramp_end_epoch - self.recon_loss_weight_ramp_start_epoch)
        self.recon_loss_weight = recon_loss_weight
        scaled_recon_loss = recon_loss * recon_loss_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mu, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)
        tc_scale = (self.tc_weight_max - self.tc_weight_min) * \
            min(1.0, (epoch_idx - self.tc_start_epoch) /
            self.tc_warmup_epochs) + self.tc_weight_min if epoch_idx > self.tc_start_epoch else self.tc_weight_min
        scaled_vae_tc_loss = vae_tc_loss * tc_scale

        # VAE total loss
        vae_loss = scaled_recon_loss + scaled_kld_loss + scaled_vae_tc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        vae_optimizer.step()
        vae_scheduler.step(vae_loss.item())

        # Discriminator forward pass
        # encode with the VAE
        self.model.eval()
        mu_2, logvar_2 = self.model.encode(x2)
        # reparameterize
        z_2 = self.model.reparameterize(mu_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        # get the discriminator output
        d_z_detached = self.D(z.detach())
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z_detached, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))

        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss)
        self.clip_gradients(d_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # D step
        d_optimizer.step()
        d_scheduler.step(d_tc_loss.item())

        # log losses
        self.last_recon_loss = recon_loss # using it for dynamic kld threshold
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
            "lr_vae": vae_scheduler.get_last_lr()[0],
            "lr_d": d_scheduler.get_last_lr()[0],
            "vae_recon_loss_weight": self.recon_loss_weight,
            "vae_kld_scale": kld_scale,
            "vae_tc_scale": tc_scale,
        },
        prog_bar=True)


    def on_train_epoch_end(self):
        # update the kld weight
        changed_kld, changed_param = False, False
        if self.args.dynamic_kld > 0:
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic *= 1.01
                changed_kld = True
        # if changed param or kld weights, reset scheduler bad epochs
        if changed_param or changed_kld:
            epoch = self.trainer.current_epoch
            vae_scheduler, d_scheduler = self.lr_schedulers()
            for scheduler in [vae_scheduler, d_scheduler]:
                # Check if this is a ReduceLROnPlateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.num_bad_epochs = 0
                    print(f"Resetting patience for {scheduler} at epoch {epoch}")


    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr_vae * lr_scale
            for pg in self.trainer.optimizers[1].param_groups:
                pg["lr"] = self.lr_d * lr_scale


    def configure_optimizers(self):
        vae_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_vae)
        d_optimizer = torch.optim.AdamW(
            self.D.parameters(), lr=self.lr_d)
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, mode='min', factor=self.lr_decay_vae, patience=20000)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, mode='min', factor=self.lr_decay_d, patience=40000)
        # return the optimizers and schedulers
        return [vae_optimizer, d_optimizer], [vae_scheduler, d_scheduler]



class PlSineFactorVAE(LightningModule):
    def __init__(self, args):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.args = args

        self.sr = args.sr
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.recon_loss_weight_start = args.recon_loss_weight_start
        self.recon_loss_weight = args.recon_loss_weight_start # initialize to start
        self.recon_loss_weight_end = args.recon_loss_weight_end
        self.recon_loss_weight_ramp_start_epoch = args.recon_loss_weight_ramp_start_epoch
        self.recon_loss_weight_ramp_end_epoch = args.recon_loss_weight_ramp_end_epoch
        self.n_samples = args.length_samps
        self.n_fft = args.n_fft
        self.latent_size = args.latent_size
        self.logdir = args.logdir
        self.d_hidden_size = args.d_hidden_size
        self.d_num_layers = args.d_num_layers

        # losses
        self.recon_loss = nn.MSELoss() if self.args.recon_loss_type == 'mse' else nn.L1Loss()
        self.recon_loss.eval()
        self.kld = kld_loss
        self.kld_weight_max = args.kld_weight_max
        self.kld_weight_min = args.kld_weight_min
        self.kld_weight_dynamic = args.kld_weight_min  # initialize to min
        self.kld_start_epoch = args.kld_start_epoch
        self.kld_warmup_epochs = args.kld_warmup_epochs
        self.tc_weight_max = args.tc_weight_max
        self.tc_weight_min = args.tc_weight_min
        self.tc_start_epoch = args.tc_start_epoch
        self.tc_warmup_epochs = args.tc_warmup_epochs

        # learning rates
        self.lr_vae = args.lr_vae
        self.lr_decay_vae = args.lr_decay_vae
        self.lr_d = args.lr_d
        self.lr_decay_d = args.lr_decay_d

        # models
        self.input_synth = Sinewave(sr=self.sr)
        self.input_synth.eval()
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            f_min=args.f_min,
            f_max=args.f_max,
            n_mels=args.n_mels,
            power=args.power,
            normalized=args.normalized > 0,
        )
        self.amplitude_to_db = AmplitudeToDB(stype='power', top_db=80.0)
        self.mel_spectrogram.eval()
        self.model = SineVAE(
            input_width=self.args.n_mels,
            encoder_channels=args.encoder_channels,
            encoder_kernels=args.encoder_kernels,
            encoder_n_res_block=args.encoder_n_res_block,
            encoder_n_res_channel=args.encoder_n_res_channel,
            decoder_channels=args.decoder_channels,
            decoder_n_res_block=args.decoder_n_res_block,
            decoder_n_res_channel=args.decoder_n_res_channel,
            latent_size=self.latent_size,
        )
        self.D = LinearDiscriminator(self.latent_size, self.d_hidden_size, 2, self.d_num_layers)
        

        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)
        self.D.apply(init_weights_kaiming)


    def sample_sine_params(self, batch_size):
        pitches_norm = torch.rand(batch_size, requires_grad=False, device=self.device)
        pitches = scale(pitches_norm, 0, 1, 38, 86)
        freqs = midi2frequency(pitches)
        freqs = freqs.unsqueeze(1).repeat(1, self.sr)
        amps = torch.rand(batch_size, requires_grad=False, device=self.device) * 0.99 + 0.01
        norm_params = torch.stack((pitches_norm, amps), dim=1)
        return norm_params, freqs, amps
    

    def forward(self, x):
        in_wf = x.unsqueeze(1)
        # get mel spectrogram
        in_spec = self.mel_spectrogram(in_wf)
        # convert to dB
        in_spec = self.amplitude_to_db(in_spec)
        # average time dimension, keep batch and mel dims
        in_spec = torch.mean(in_spec, dim=-1)
        # scale by bin minmax
        in_spec = scale(in_spec, self.args.bin_minmax[0], self.args.bin_minmax[1], 0, 1)
        # predict
        out_spec, mu, logvar, z = self.model(in_spec)
        return out_spec, mu, logvar, z
    

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.D.train()

        # get the optimizers and schedulers
        vae_optimizer, d_optimizer = self.optimizers()
        vae_scheduler, d_scheduler = self.lr_schedulers()

        # create a batch of ones and zeros for the discriminator
        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # get the batch
        epoch_idx = self.trainer.current_epoch

        norm_params, freqs, amps = self.sample_sine_params(self.batch_size)
        x = self.input_synth(freqs).detach() * amps.unsqueeze(1)
        # select a random slice of self.n_samples
        start_idx = torch.randint(0, self.sr - self.n_samples, (1,))
        x = x[:, start_idx:start_idx + self.n_samples]
        # add random phase flip
        phase_flip = torch.rand(self.batch_size, 1, device=self.device)
        phase_flip = torch.where(phase_flip > 0.5, 1, -1)
        x = x * phase_flip
        in_wf_slice = x.unsqueeze(1)

        # forward pass
        # get mel spectrogram
        in_spec = self.mel_spectrogram(in_wf_slice.detach())
        # convert to dB
        in_spec = self.amplitude_to_db(in_spec)
        # average time dimension, keep batch and mel dims
        in_spec = torch.mean(in_spec, dim=-1)
        # scale by bin minmax
        in_spec = scale(in_spec, self.args.bin_minmax[0], self.args.bin_minmax[1], 0, 1)
        # predict
        out_spec, mu, logvar, z = self.model(in_spec)

        # VAE recon_loss
        recon_loss = self.recon_loss(out_spec, in_spec)
        # calculate current recon loss weight
        current_epoch = self.trainer.current_epoch
        if current_epoch < self.recon_loss_weight_ramp_start_epoch:
            recon_loss_weight = self.recon_loss_weight_start
        elif current_epoch > self.recon_loss_weight_ramp_end_epoch:
            recon_loss_weight = self.recon_loss_weight_end
        else:
            recon_loss_weight = self.recon_loss_weight_start + (self.recon_loss_weight_end - self.recon_loss_weight_start) * \
                (current_epoch - self.recon_loss_weight_ramp_start_epoch) / \
                (self.recon_loss_weight_ramp_end_epoch - self.recon_loss_weight_ramp_start_epoch)
        self.recon_loss_weight = recon_loss_weight
        scaled_recon_loss = recon_loss * recon_loss_weight

        # VAE KLD loss
        if self.args.dynamic_kld > 0:
            kld_scale = self.kld_weight_dynamic
        else:
            kld_scale = (self.kld_weight_max - self.kld_weight_min) * \
                min(1.0, (epoch_idx - self.kld_start_epoch) /
                    self.kld_warmup_epochs) + self.kld_weight_min if epoch_idx > self.kld_start_epoch else self.kld_weight_min
        kld_loss = self.kld(mu, logvar)
        scaled_kld_loss = kld_loss * kld_scale

        # VAE TC loss
        d_z = self.D(z)
        vae_tc_loss = F.cross_entropy(d_z, ones)
        tc_scale = (self.tc_weight_max - self.tc_weight_min) * \
            min(1.0, (epoch_idx - self.tc_start_epoch) /
            self.tc_warmup_epochs) + self.tc_weight_min if epoch_idx > self.tc_start_epoch else self.tc_weight_min
        scaled_vae_tc_loss = vae_tc_loss * tc_scale

        # VAE total loss
        vae_loss = scaled_recon_loss + scaled_kld_loss + scaled_vae_tc_loss

        # VAE backward pass
        vae_optimizer.zero_grad()
        self.manual_backward(vae_loss, retain_graph=True)
        self.clip_gradients(vae_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        vae_optimizer.step()
        vae_scheduler.step(vae_loss.item())

        # get another batch for D
        norm_params, freqs, amps = self.sample_sine_params(self.batch_size)
        x = self.input_synth(freqs).detach() * amps.unsqueeze(1)
        in_wf = x.unsqueeze(1)
        # select a random slice of self.n_samples
        start_idx = torch.randint(0, self.sr - self.n_samples, (1,))
        x = x[:, start_idx:start_idx + self.n_samples]
        # add random phase flip
        phase_flip = torch.rand(self.batch_size, 1, device=self.device)
        phase_flip = torch.where(phase_flip > 0.5, 1, -1)
        x = x * phase_flip
        in_wf_slice = x.unsqueeze(1)
        # get mel spectrogram
        in_spec = self.mel_spectrogram(in_wf_slice.detach())
        # convert to dB
        in_spec = self.amplitude_to_db(in_spec)
        # average time dimension, keep batch and mel dims
        in_spec = torch.mean(in_spec, dim=-1)
        # scale by bin minmax
        in_spec = scale(in_spec, self.args.bin_minmax[0], self.args.bin_minmax[1], 0, 1)
        # encode with the VAE
        self.model.eval()
        mu_2, logvar_2 = self.model.encode(in_spec)
        # reparameterize
        z_2 = self.model.reparameterize(mu_2, logvar_2)
        z_2_perm = permute_dims(z_2)
        # get the discriminator output
        d_z_detached = self.D(z.detach())
        d_z_2_perm = self.D(z_2_perm.detach())
        d_tc_loss = 0.5 * (F.cross_entropy(d_z_detached, zeros) +
                           F.cross_entropy(d_z_2_perm, ones))
        
        # Discriminator backward pass
        d_optimizer.zero_grad()
        self.manual_backward(d_tc_loss)
        self.clip_gradients(d_optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # D step
        d_optimizer.step()
        d_scheduler.step(d_tc_loss.item())

        # log losses
        self.last_recon_loss = recon_loss # using it for dynamic kld threshold
        self.log_dict({
            "vae_loss": vae_loss,
            "vae_recon_loss": recon_loss,
            "vae_kld_loss": kld_loss,
            "vae_tc_loss": vae_tc_loss,
            "d_tc_loss": d_tc_loss,
            "lr_vae": vae_scheduler.get_last_lr()[0],
            "lr_d": d_scheduler.get_last_lr()[0],
            "vae_recon_loss_weight": self.recon_loss_weight,
            "vae_kld_scale": kld_scale,
            "vae_tc_scale": tc_scale,
        },
        prog_bar=True)


    def on_train_epoch_end(self):
        # update the kld weight
        changed_kld, changed_param = False, False
        if self.args.dynamic_kld > 0:
            if self.last_recon_loss < self.args.target_recon_loss:
                self.kld_weight_dynamic *= 1.01
                changed_kld = True
        # if changed param or kld weights, reset scheduler bad epochs
        if changed_param or changed_kld:
            epoch = self.trainer.current_epoch
            vae_scheduler, d_scheduler = self.lr_schedulers()
            for scheduler in [vae_scheduler, d_scheduler]:
                # Check if this is a ReduceLROnPlateau scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.num_bad_epochs = 0
                    print(f"Resetting patience for {scheduler} at epoch {epoch}")


    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr_vae * lr_scale
            for pg in self.trainer.optimizers[1].param_groups:
                pg["lr"] = self.lr_d * lr_scale


    def configure_optimizers(self):
        vae_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr_vae)
        d_optimizer = torch.optim.AdamW(
            self.D.parameters(), lr=self.lr_d)
        vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer, mode='min', factor=self.lr_decay_vae, patience=20000)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            d_optimizer, mode='min', factor=self.lr_decay_d, patience=40000)
        # return the optimizers and schedulers
        return [vae_optimizer, d_optimizer], [vae_scheduler, d_scheduler]


class PlMapper(LightningModule):
    def __init__(self, args):
        super(PlMapper, self).__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.args = args

        self.in_features = args.in_features
        self.out_features = args.out_features
        self.hidden_layers_features = args.hidden_layers_features
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs

        # losses
        self.locality_loss = nn.MSELoss() if self.args.locality_loss_type == "mse" else nn.L1Loss()
        self.locality_weight = args.locality_weight
        self.cycle_consistency_loss = nn.MSELoss() if self.args.cycle_consistency_loss_type == "mse" else nn.L1Loss()
        self.cycle_consistency_weight = args.cycle_consistency_weight_start # initialize to start
        self.cycle_consistency_weight_start = args.cycle_consistency_weight_start
        self.cycle_consistency_weight_end = args.cycle_consistency_weight_end
        self.cycle_consistency_ramp_start_epoch = args.cycle_consistency_ramp_start_epoch
        self.cycle_consistency_ramp_end_epoch = args.cycle_consistency_ramp_end_epoch

        # learning rate
        self.lr = args.lr
        self.lr_decay = args.lr_decay

        # models
        self.model = LinearProjector(in_features=self.in_features, out_features=self.out_features, hidden_layers_features=self.hidden_layers_features)
        self.in_model = args.in_model
        self.out_model = args.out_model
        self.in_model.eval()
        self.out_model.eval()

        def init_weights_kaiming(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):  # Apply to conv and linear layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.model.apply(init_weights_kaiming)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.in_model.eval()
        self.out_model.eval()

        # get the optimizers and schedulers
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        # get the batch
        epoch_idx = self.trainer.current_epoch
        x, x2 = batch

        # encode with input model
        mu, logvar = self.in_model.model.encode(x)
        z_1 = self.in_model.model.reparameterize(mu, logvar)

        # project to output space
        z_2 = self.model(z_1)

        # decode with output model
        x_hat = self.out_model.model.decoder(z_2)

        # re-encode with output model
        mu, logvar = self.out_model.model.encode(x_hat.detach())
        z_3 = self.out_model.model.reparameterize(mu, logvar)

        # LOSSES
        # locality loss
        num_dims = z_1.shape[1]
        locality_loss = 0
        for dim in range(num_dims):
            locality_loss += self.locality_loss(z_2[:, dim], z_1[:, dim])
        scaled_locality_loss = self.locality_weight * locality_loss

        # cycle consistency loss
        cycle_consistency_loss = 0
        # calculate current cycle consistency loss weight
        if epoch_idx < self.cycle_consistency_ramp_start_epoch:
            cycle_consistency_weight = self.cycle_consistency_weight_start
        elif epoch_idx > self.cycle_consistency_ramp_end_epoch:
            cycle_consistency_weight = self.cycle_consistency_weight_end
        else:
            cycle_consistency_weight = self.cycle_consistency_weight_start + (self.cycle_consistency_weight_end - self.cycle_consistency_weight_start) * \
                (epoch_idx - self.cycle_consistency_ramp_start_epoch) / \
                (self.cycle_consistency_ramp_end_epoch - self.cycle_consistency_ramp_start_epoch)
        self.cycle_consistency_weight = cycle_consistency_weight
        if cycle_consistency_weight > 0:
            cycle_consistency_loss = self.cycle_consistency_loss(z_3, z_2)
        scaled_cycle_consistency_loss = cycle_consistency_weight * cycle_consistency_loss

        # total loss
        loss = scaled_locality_loss + scaled_cycle_consistency_loss

        # backward pass
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step(loss.item())

        # log losses
        self.log_dict({
            "loss": loss,
            "locality_loss": locality_loss,
            "cycle_consistency_loss": cycle_consistency_loss,
            "cycle_consistency_scale": self.cycle_consistency_weight,
            "lr": scheduler.get_last_lr()[0],
        }, prog_bar=True)

    def on_train_batch_start(self, batch, batch_idx):
        epoch = self.trainer.current_epoch
        if epoch < self.warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / self.warmup_epochs)
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = self.lr * lr_scale

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.lr_decay, patience=20000)
        return [optimizer], [scheduler]
