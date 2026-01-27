# -*- coding: utf-8 -*-
"""
@author: iremc
"""
import torch
from torch.nn import functional as F
import torch.nn as nn

# For the Unet model
from torch.utils.checkpoint import checkpoint
from IPython import embed
from torch import Tensor
import numpy as np

def initialize_weights(m):
    """Weight initialization for ConvVAE (Conv3d-based)."""
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def initialize_weights_unet(m):
    """
    Weight initialization for UNetVAE (Conv2d-based with GroupNorm and Attention).

    Uses:
    - Xavier/Glorot uniform for Conv2d and Linear layers
    - Ones/zeros for normalization layers (GroupNorm, BatchNorm1d)
    - Default PyTorch init for MultiheadAttention (already well-initialized)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.MultiheadAttention):
        # MultiheadAttention has in_proj_weight and out_proj
        if m.in_proj_weight is not None:
            nn.init.xavier_uniform_(m.in_proj_weight)
        if m.out_proj.weight is not None:
            nn.init.xavier_uniform_(m.out_proj.weight)
        if m.in_proj_bias is not None:
            nn.init.constant_(m.in_proj_bias, 0)
        if m.out_proj.bias is not None:
            nn.init.constant_(m.out_proj.bias, 0)
     
unflatten_channel = 1
dim_start_up_decoder = [16,4,4]
class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class Unflatten(nn.Module):
     def forward(self, input, size= unflatten_channel):
        return input.view(input.size(0), dim_start_up_decoder[0], dim_start_up_decoder[1], dim_start_up_decoder[2])

class ConvVAE(nn.Module):

    def __init__(self, image_channels, h_dim, latent_size, n_filters_ENC, n_filters_DEC):
        super(ConvVAE, self).__init__()

        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_size = latent_size
        self.n_filters_ENC = n_filters_ENC
        self.n_filters_DEC = n_filters_DEC
        
    ##############
    ## ENCODER ##
    ##############
        self.conv1_enc = nn.Conv3d(in_channels = image_channels, out_channels = n_filters_ENC[0],  kernel_size = [3,3,3] , stride = 2, padding =1 )
        self.bn1_enc = nn.BatchNorm3d( n_filters_ENC[0])
        self.conv2_enc = nn.Conv3d(in_channels = n_filters_ENC[0], out_channels = n_filters_ENC[1], kernel_size = [3,3,3] , stride = 2, padding =1)
        self.bn2_enc = nn.BatchNorm3d( n_filters_ENC[1])
        self.conv3_enc = nn.Conv3d(in_channels = n_filters_ENC[1], out_channels = n_filters_ENC[2],  kernel_size = [3,3,3] , stride = 2, padding =1)
        self.bn3_enc = nn.BatchNorm3d( n_filters_ENC[2])
        self.conv4_enc = nn.Conv3d(in_channels = n_filters_ENC[2], out_channels = n_filters_ENC[3],  kernel_size = [3,3,3] , stride = 2, padding =1)
        self.bn4_enc = nn.BatchNorm3d( n_filters_ENC[3])
        self.conv5_enc = nn.Conv3d(in_channels = n_filters_ENC[3], out_channels = n_filters_ENC[4], kernel_size = [3,3,3], stride = 1 , padding =1)
        self.bn5_enc = nn.BatchNorm3d( n_filters_ENC[4])
        
        self.flatten = Flatten() 
             
        self.fc1 = nn.Linear(250, 128)     

        self.fc2 = nn.Linear(128, h_dim)
              
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

        # icetin: hidden => mu
        self.mu = nn.Linear(h_dim, latent_size)
        # icetin: hidden => logvar
        self.logvar = nn.Linear(h_dim, latent_size)# icetin: same above

        # icetin: MLP
        self.mlp1 = nn.Linear(latent_size, int(latent_size/2))
        self.bn1_mlp = nn.BatchNorm1d(int(latent_size/2))
        self.mlp2 = nn.Linear(int(latent_size/2), int(latent_size/4)) 
        self.bn2_mlp = nn.BatchNorm1d(int(latent_size/4))
        self.mlp3 = nn.Linear(int(latent_size/4), 1) 
        #self.bn3_mlp = nn.BatchNorm1d(1)
        self.sigmoid_mlp = nn.Sigmoid()
    ###################
    ### END OF ENCODER
    ###################
      
    ###############
    ### DECODER 
    ###############
        #icetin: biffi et. al decoder, LVAE + MLP
        self.fc3 = nn.Linear(latent_size, 250) #icetin: pulls from bottleneck to hidden # dim_start_up_decoder = [5,5,5]
        self.unflatten = Unflatten()
        
        
        self.conv1_dec = nn.Conv3d(in_channels = unflatten_channel, out_channels = n_filters_DEC[0],  kernel_size = [3,3,3], stride = 1, padding =1)
        self.bn1_dec = nn.BatchNorm3d(n_filters_DEC[0])
        self.deconv1_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[0], out_channels =  n_filters_DEC[0], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn2_dec = nn.BatchNorm3d(n_filters_DEC[0])
        self.deconv2_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[0], out_channels =  n_filters_DEC[1], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn3_dec = nn.BatchNorm3d(n_filters_DEC[1])
        self.deconv3_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[1], out_channels =  n_filters_DEC[2], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn4_dec = nn.BatchNorm3d(n_filters_DEC[2])
        self.deconv4_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[2], out_channels =  n_filters_DEC[3], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn5_dec = nn.BatchNorm3d(n_filters_DEC[3])
        self.conv2_dec = nn.Conv3d(in_channels = n_filters_DEC[3], out_channels = n_filters_DEC[4], kernel_size = [3,3,3], stride = 1, padding =1)
        self.bn6_dec = nn.BatchNorm3d(n_filters_DEC[4])
        self.conv3_dec = nn.Conv3d(in_channels = n_filters_DEC[4], out_channels = image_channels, kernel_size = [3,3,3], stride = 1, padding =1)
        self.bn7_dec = nn.BatchNorm3d(image_channels)
        

        self.sigmoid = nn.Sigmoid() # No need : sigmoid is used in the loss - when to set 'gaussian'
        #self.tanh = nn.Tanh()
   ##################
   ### END OF DECODER
   ##################

    def encode(self, x): # encoder returns mu and logvar
        
        h = F.relu(self.bn1_enc(self.conv1_enc(x)))
        h = F.relu(self.bn2_enc(self.conv2_enc(h)))
        h = F.relu(self.bn3_enc(self.conv3_enc(h)))
        h = F.relu(self.bn4_enc(self.conv4_enc(h)))
        h = F.relu(self.bn5_enc(self.conv5_enc(h)))

        h = self.dropout(self.flatten(h))
        
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        
        mu, logvar = self.mu(h), self.logvar(h)

        ####define the distribution from mu and logvar
        # FIX: logvar = log(variance) = log(σ²), so std = σ = exp(0.5 * logvar)
        # Previously used exp(logvar) = σ² as scale, which caused KL divergence to explode
        z_distribution = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))
        return mu, logvar, z_distribution

    def decode(self, z): # input of the decoder is z and returns reconstructed image.
        z = F.relu(self.fc3(z))
        z = self.unflatten(z) # 

        z = F.relu(self.bn1_dec(self.conv1_dec(z)))
        z = F.relu(self.bn2_dec(self.deconv1_dec(z)))
        z = F.relu(self.bn3_dec(self.deconv2_dec(z)))
        z = F.relu(self.bn4_dec(self.deconv3_dec(z)))
        z = F.relu(self.bn5_dec(self.deconv4_dec(z)))
        z = F.relu(self.bn6_dec(self.conv2_dec(z)))
        z = self.conv3_dec(z)

        z = self.sigmoid(z)
        
        return z

    def mlp_predict(self, z): #icetin: mlp part that is connected to z
        out_mlp = F.relu(self.bn1_mlp(self.mlp1(z))) # input: z output: prediction
        out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
        out_mlp = self.mlp3(out_mlp)  # raw logits (no sigmoid - CrossEntropyLoss handles softmax)
        return out_mlp

    def reparameterize(self, mu, logvar, z_dist):
        """
        Reparameterization trick for VAE.

        NOTE on redundancy: z_sampled_eq and z_tilde are equivalent after the fix to z_dist.
        - z_sampled_eq: manual reparameterization (mu + std * epsilon)
        - z_tilde: PyTorch's rsample() which does the same thing internally
        Both produce differentiable samples from the posterior q(z|x).

        Why this function is still useful:
        - Creates prior_dist N(0, I) needed for KL divergence computation
        - Provides a single place to handle all sampling logic
        """
        # Manual reparameterization trick (redundant with z_tilde, kept for reference)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sampled_eq = eps.mul(std).add_(mu)

        # Compute prior: standard normal distribution N(0, I)
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # Sample from posterior using PyTorch's rsample (differentiable via reparameterization)
        z_tilde = z_dist.rsample()
        return z_tilde, z_sampled_eq, z_prior, prior_dist
 
    def reparameterize_eval(self, mu, logvar):
        #print("REPARAMETERIZE EVAL...")
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def z_return(self, z):
        return z

    def forward(self, x): # forward prop of the network.
        mu, logvar, z_dist = self.encode(x) # encoder returns mu and sigma
        
        z_tilde, z_sampled_eq, z_prior, prior_dist  = self.reparameterize(mu, logvar, z_dist) # reparameterization trick returns sample, z
        out_mlp = self.mlp_predict(z_tilde) # mlp branch takes z and outputs the predictions

        output = self.decode(z_tilde) # before z_sampled_eq was inputted
        return output, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist # reconstructed x, mu, logvar, mlp output
    

### DEFINE ConvVAE2D (adapted from ConvVAE for 2D images) ###

def initialize_weights_2d(m):
    """Weight initialization for ConvVAE2D."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class ConvVAE2D(nn.Module):
    """
    2D Convolutional VAE adapted from the original 3D ConvVAE.

    This is a simple encoder-decoder architecture WITHOUT skip connections,
    suitable for latent space compression and controllable generation.

    Architecture:
        Encoder: Conv2d (stride 2) x4 → Flatten → FC → μ, σ
        Decoder: FC → Unflatten → ConvTranspose2d (stride 2) x4 → Output
    """

    def __init__(self, image_channels, h_dim, latent_size, n_filters_ENC, n_filters_DEC,
                 img_size=64, num_classes=3):
        super(ConvVAE2D, self).__init__()

        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_size = latent_size
        self.n_filters_ENC = n_filters_ENC
        self.n_filters_DEC = n_filters_DEC
        self.img_size = img_size
        self.num_classes = num_classes

        # Compute spatial size after encoder (4 stride-2 convolutions)
        # 64 -> 32 -> 16 -> 8 -> 4
        self.bottleneck_spatial = img_size // 16  # 4 for 64x64 input

        ##############
        ## ENCODER ##
        ##############
        self.conv1_enc = nn.Conv2d(image_channels, n_filters_ENC[0], kernel_size=3, stride=2, padding=1)
        self.bn1_enc = nn.BatchNorm2d(n_filters_ENC[0])

        self.conv2_enc = nn.Conv2d(n_filters_ENC[0], n_filters_ENC[1], kernel_size=3, stride=2, padding=1)
        self.bn2_enc = nn.BatchNorm2d(n_filters_ENC[1])

        self.conv3_enc = nn.Conv2d(n_filters_ENC[1], n_filters_ENC[2], kernel_size=3, stride=2, padding=1)
        self.bn3_enc = nn.BatchNorm2d(n_filters_ENC[2])

        self.conv4_enc = nn.Conv2d(n_filters_ENC[2], n_filters_ENC[3], kernel_size=3, stride=2, padding=1)
        self.bn4_enc = nn.BatchNorm2d(n_filters_ENC[3])

        self.conv5_enc = nn.Conv2d(n_filters_ENC[3], n_filters_ENC[4], kernel_size=3, stride=1, padding=1)
        self.bn5_enc = nn.BatchNorm2d(n_filters_ENC[4])

        # Compute flattened size: channels * spatial^2
        self.flattened_size = n_filters_ENC[4] * self.bottleneck_spatial * self.bottleneck_spatial

        # FC layers for encoder
        self.fc1 = nn.Linear(self.flattened_size, h_dim * 2)
        self.fc2 = nn.Linear(h_dim * 2, h_dim)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Latent space
        self.mu = nn.Linear(h_dim, latent_size)
        self.logvar = nn.Linear(h_dim, latent_size)

        # MLP classifier (for attri-VAE)
        self.mlp1 = nn.Linear(latent_size, latent_size // 2)
        self.bn1_mlp = nn.BatchNorm1d(latent_size // 2)
        self.mlp2 = nn.Linear(latent_size // 2, latent_size // 4)
        self.bn2_mlp = nn.BatchNorm1d(latent_size // 4)
        self.mlp3 = nn.Linear(latent_size // 4, num_classes)

        ###############
        ### DECODER ###
        ###############

        # Decoder channels (reverse of encoder)
        self.decoder_init_channels = n_filters_ENC[4]

        self.fc3 = nn.Linear(latent_size, self.flattened_size)

        self.conv1_dec = nn.Conv2d(n_filters_ENC[4], n_filters_DEC[0], kernel_size=3, stride=1, padding=1)
        self.bn1_dec = nn.BatchNorm2d(n_filters_DEC[0])

        self.deconv1_dec = nn.ConvTranspose2d(n_filters_DEC[0], n_filters_DEC[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2_dec = nn.BatchNorm2d(n_filters_DEC[0])

        self.deconv2_dec = nn.ConvTranspose2d(n_filters_DEC[0], n_filters_DEC[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3_dec = nn.BatchNorm2d(n_filters_DEC[1])

        self.deconv3_dec = nn.ConvTranspose2d(n_filters_DEC[1], n_filters_DEC[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4_dec = nn.BatchNorm2d(n_filters_DEC[2])

        self.deconv4_dec = nn.ConvTranspose2d(n_filters_DEC[2], n_filters_DEC[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5_dec = nn.BatchNorm2d(n_filters_DEC[3])

        self.conv2_dec = nn.Conv2d(n_filters_DEC[3], n_filters_DEC[4], kernel_size=3, stride=1, padding=1)
        self.bn6_dec = nn.BatchNorm2d(n_filters_DEC[4])

        self.conv3_dec = nn.Conv2d(n_filters_DEC[4], image_channels, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """Encoder: image -> mu, logvar, z_distribution"""
        h = F.relu(self.bn1_enc(self.conv1_enc(x)))
        h = F.relu(self.bn2_enc(self.conv2_enc(h)))
        h = F.relu(self.bn3_enc(self.conv3_enc(h)))
        h = F.relu(self.bn4_enc(self.conv4_enc(h)))
        h = F.relu(self.bn5_enc(self.conv5_enc(h)))

        # Flatten
        h = h.view(h.size(0), -1)
        h = self.dropout(h)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        mu, logvar = self.mu(h), self.logvar(h)

        # Define the distribution
        z_distribution = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))
        return mu, logvar, z_distribution

    def decode(self, z):
        """Decoder: z -> reconstructed image"""
        z = F.relu(self.fc3(z))
        z = z.view(-1, self.decoder_init_channels, self.bottleneck_spatial, self.bottleneck_spatial)

        z = F.relu(self.bn1_dec(self.conv1_dec(z)))
        z = F.relu(self.bn2_dec(self.deconv1_dec(z)))
        z = F.relu(self.bn3_dec(self.deconv2_dec(z)))
        z = F.relu(self.bn4_dec(self.deconv3_dec(z)))
        z = F.relu(self.bn5_dec(self.deconv4_dec(z)))
        z = F.relu(self.bn6_dec(self.conv2_dec(z)))
        z = self.conv3_dec(z)

        z = self.sigmoid(z)

        return z

    def mlp_predict(self, z):
        """MLP classifier for attri-VAE"""
        out_mlp = F.relu(self.bn1_mlp(self.mlp1(z)))
        out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
        out_mlp = self.mlp3(out_mlp)  # raw logits
        return out_mlp

    def reparameterize(self, mu, logvar, z_dist):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sampled_eq = eps.mul(std).add_(mu)

        # Prior distribution N(0, I)
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # Sample from posterior
        z_tilde = z_dist.rsample()
        return z_tilde, z_sampled_eq, z_prior, prior_dist

    def forward(self, x):
        """Forward pass: x -> reconstruction, mu, logvar, mlp_output, etc."""
        mu, logvar, z_dist = self.encode(x)

        z_tilde, z_sampled_eq, z_prior, prior_dist = self.reparameterize(mu, logvar, z_dist)
        out_mlp = self.mlp_predict(z_tilde)

        output = self.decode(z_tilde)
        return output, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist


### DEFINE UNET VAE ###

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    """
    Pre Layer norm  -> multi-headed tension -> skip connections -> pass it to
    the feed forward layer (layer-norm -> 2 multiheadattention)
    """
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """
    Normal convolution block, with 2d convolution -> Group Norm -> GeLU -> convolution -> Group Norm
    Possibility to add residual connection providing residual=True
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    maxpool reduce size by half -> 2*DoubleConv -> Embedding layer
    
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """
    We take the skip connection which comes from the encoder
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x 
    
class UNetVAE(nn.Module):
    def __init__(self, image_channels, h_dim, latent_size, n_filters_ENC, n_filters_DEC, img_size, device='cuda',
                 skip_mode='full', skip_dropout_prob=0.5, skip_scale=0.3):
        """
        UNet-based VAE with configurable skip connections.

        Args:
            skip_mode: 'full' = use all skip connections (default, best reconstruction)
                       'none' = no skip connections (best for latent manipulation)
                       'dropout' = randomly drop skip connections during training
                       'weighted' = scale skip connections by skip_scale (balance control vs quality)
            skip_dropout_prob: probability of dropping each skip connection (only used if skip_mode='dropout')
            skip_scale: scale factor for 'weighted' mode (0=no skips/max control, 1=full skips/best quality)
        """
        super(UNetVAE, self).__init__()

        # Encoder
        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_size = latent_size
        self.n_filters_ENC = n_filters_ENC
        self.n_filters_DEC = n_filters_DEC
        self.img_size = img_size
        self.device = device
        self.skip_mode = skip_mode
        self.skip_dropout_prob = skip_dropout_prob
        self.skip_scale = skip_scale

        self.inc = DoubleConv(self.image_channels, self.n_filters_ENC[0]) # Wrap-up for 2 Conv Layers
        self.down1 = Down(self.n_filters_ENC[0], self.n_filters_ENC[1]) # input and output channels
        self.sa1 = SelfAttention(self.n_filters_ENC[1], int(self.img_size // 2)) # 1st is channel dim, 2nd current image resolution
        self.down2 = Down(self.n_filters_ENC[1], self.n_filters_ENC[2])
        self.sa2 = SelfAttention(self.n_filters_ENC[2], int(self.img_size // 4))
        self.down3 = Down(self.n_filters_ENC[2], self.n_filters_ENC[3])
        self.sa3 = SelfAttention(self.n_filters_ENC[3], int(self.img_size // 8))
        self.down4 = Down(self.n_filters_ENC[3], self.n_filters_ENC[4])
        self.sa4 = SelfAttention(self.n_filters_ENC[4], int(self.img_size // 16))

        self.flatten = Flatten()

        # Compute flattened size dynamically: channels * spatial_size^2
        # After 4 downsampling stages (each halves spatial dim): img_size // 16
        self.bottleneck_spatial = self.img_size // 16
        self.flattened_size = self.n_filters_ENC[4] * self.bottleneck_spatial * self.bottleneck_spatial

        # Intermediate size: gradual compression from flattened_size to h_dim
        # Use midpoint or at least h_dim to avoid unnecessary bottleneck
        self.fc_intermediate = max(self.flattened_size // 2, h_dim) # BEFORE IT WAS 128 

        self.fc1 = nn.Linear(self.flattened_size, self.fc_intermediate)
        self.fc2 = nn.Linear(self.fc_intermediate, h_dim)
        
        ## Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

        # icetin: hidden => mu
        self.mu = nn.Linear(h_dim, latent_size)
        # icetin: hidden => logvar
        self.logvar = nn.Linear(h_dim, latent_size)# icetin: same above

        # icetin: MLP (3-class classification: low=0, medium=1, high=2)
        self.num_classes = 3
        self.mlp1 = nn.Linear(latent_size, int(latent_size/2))
        self.bn1_mlp = nn.BatchNorm1d(int(latent_size/2))
        self.mlp2 = nn.Linear(int(latent_size/2), int(latent_size/4))
        self.bn2_mlp = nn.BatchNorm1d(int(latent_size/4))
        self.mlp3 = nn.Linear(int(latent_size/4), self.num_classes)  # 3 classes output (logits)

        ###################
        ### END OF ENCODER
        ###################

        ###############
        ### DECODER 
        ###############

        #icetin: biffi et. al decoder, LVAE + MLP
        self.fc3 = nn.Linear(latent_size, self.flattened_size)
        # Unflatten back to (n_filters_ENC[4], bottleneck_spatial, bottleneck_spatial)
        self.unflatten_shape = (self.n_filters_ENC[4], self.bottleneck_spatial, self.bottleneck_spatial)

        self.up1 = Up(self.n_filters_ENC[4] + self.n_filters_ENC[3], self.n_filters_ENC[3])
        self.sa5 = SelfAttention(self.n_filters_ENC[3], int(self.img_size // 8))
        self.up2 = Up(self.n_filters_ENC[3] + self.n_filters_ENC[2], self.n_filters_ENC[2])
        self.sa6 = SelfAttention(self.n_filters_ENC[2], int(self.img_size // 4))
        self.up3 = Up(self.n_filters_ENC[2] + self.n_filters_ENC[1], self.n_filters_ENC[1])
        self.sa7 = SelfAttention(self.n_filters_ENC[1], int(self.img_size // 2))
        self.up4 = Up(self.n_filters_ENC[1] + self.n_filters_ENC[0], self.n_filters_ENC[0])
        self.sa8 = SelfAttention(self.n_filters_ENC[0], int(self.img_size))
        self.outc = nn.Conv2d(self.n_filters_ENC[0], self.image_channels, kernel_size=1) # projecting back to the output channel dimensions

        self.sigmoid = nn.Sigmoid() # No need : sigmoid is used in the loss - when to set 'gaussian'


    def encode(self, x): # encoder returns mu and logvar
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)
        x5 = self.down4(x4)
        x5 = self.sa4(x5)

        h = self.dropout(self.flatten(x5))
        
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        
        mu, logvar = self.mu(h), self.logvar(h)

        ####define the distribution from mu and logvar
        # FIX: logvar = log(variance) = log(σ²), so std = σ = exp(0.5 * logvar)
        # Previously used exp(logvar) = σ² as scale, which caused KL divergence to explode
        z_distribution = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))
        return mu, logvar, z_distribution, x1, x2, x3, x4

    def _apply_skip_dropout(self, skip_tensor):
        """
        Apply skip connection modification based on skip_mode.

        Modes:
            - 'full': use skip connections as-is
            - 'none': return zeros (no skip connections)
            - 'dropout': randomly drop skip connections during training
            - 'weighted': scale skip connections by skip_scale factor
        """
        if self.skip_mode == 'none':
            # No skip connections - always return zeros
            return torch.zeros_like(skip_tensor)
        elif self.skip_mode == 'dropout' and self.training:
            # During training with dropout mode, randomly drop skip connections
            if torch.rand(1).item() < self.skip_dropout_prob:
                return torch.zeros_like(skip_tensor)
        elif self.skip_mode == 'weighted':
            # Scale skip connections: 0 = no skips (max control), 1 = full skips (best quality)
            return skip_tensor * self.skip_scale
        # 'full' mode or eval mode with dropout - use original skip connection
        return skip_tensor

    def set_skip_scale(self, scale):
        """
        Set the skip connection scale (only affects 'weighted' mode).

        Args:
            scale: float between 0 and 1
                   0 = no skip connections (maximum latent control)
                   1 = full skip connections (best reconstruction quality)
        """
        self.skip_scale = scale

    def decode(self, z, x1, x2, x3, x4): # input of the decoder is z and returns reconstructed image.
        z = F.relu(self.fc3(z))
        z = z.view(-1, *self.unflatten_shape)  # Dynamic unflatten

        # Apply skip connection dropout/masking based on skip_mode
        x1_skip = self._apply_skip_dropout(x1)
        x2_skip = self._apply_skip_dropout(x2)
        x3_skip = self._apply_skip_dropout(x3)
        x4_skip = self._apply_skip_dropout(x4)

        z1 = self.up1(z, x4_skip)
        z1 = self.sa5(z1)
        z2 = self.up2(z1, x3_skip)
        z2 = self.sa6(z2)
        z3 = self.up3(z2, x2_skip)
        z3 = self.sa7(z3)
        z4 = self.up4(z3, x1_skip)
        z4 = self.sa8(z4)
        z = self.outc(z4)

        z = self.sigmoid(z)

        return z
    
    def mlp_predict(self, z): #icetin: mlp part that is connected to z
        out_mlp = F.relu(self.bn1_mlp(self.mlp1(z))) # input: z output: prediction
        out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
        out_mlp = self.mlp3(out_mlp)  # raw logits (no sigmoid - CrossEntropyLoss handles softmax)
        return out_mlp
    
    def reparameterize(self, mu, logvar, z_dist):
        """
        Reparameterization trick for VAE.

        NOTE on redundancy: z_sampled_eq and z_tilde are equivalent after the fix to z_dist.
        - z_sampled_eq: manual reparameterization (mu + std * epsilon)
        - z_tilde: PyTorch's rsample() which does the same thing internally
        Both produce differentiable samples from the posterior q(z|x).

        Why this function is still useful:
        - Creates prior_dist N(0, I) needed for KL divergence computation
        - Provides a single place to handle all sampling logic
        """
        # Manual reparameterization trick (redundant with z_tilde, kept for reference)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sampled_eq = eps.mul(std).add_(mu)

        # Compute prior: standard normal distribution N(0, I)
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # Sample from posterior using PyTorch's rsample (differentiable via reparameterization)
        z_tilde = z_dist.rsample()
        return z_tilde, z_sampled_eq, z_prior, prior_dist
    
    def reparametrize_eval(self, mu, logvar):
        #print("REPARAMETERIZE EVAL...")
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def z_return(self, z):
        return z

    def forward(self, x): # forward prop of the network.
        mu, logvar, z_dist, x1, x2, x3, x4 = self.encode(x) # encoder returns mu and sigma
        
        z_tilde, z_sampled_eq, z_prior, prior_dist  = self.reparameterize(mu, logvar, z_dist) # reparameterization trick returns sample, z
        out_mlp = self.mlp_predict(z_tilde) # mlp branch takes z and outputs the predictions

        output = self.decode(z_tilde, x1, x2, x3, x4) # before z_sampled_eq was inputted
        return output, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist # reconstructed x, mu, logvar, mlp output
