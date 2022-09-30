# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import copy
import math

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from torch import nn
import torch.nn.functional as F


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------

@misc.profiled_function
def modulated_fc(
        x,  # Input tensor of shape [batch_size, n_feature, in_channels].
        weight,  # Weight tensor of shape [out_channels, in_channels].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        demodulate=True,  # Apply weight demodulation?
):
    batch_size = x.shape[0]
    n_feature = x.shape[1]
    out_channels, in_channels = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels])
    misc.assert_shape(x, [batch_size, n_feature, in_channels])
    misc.assert_shape(styles, [batch_size, in_channels])
    assert demodulate
    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels) / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOI]
    w = w * styles.unsqueeze(dim=1)  # [NOI]
    dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt()  # [NO]
    w = w * dcoefs.unsqueeze(dim=-1)  # [NOI]
    x = torch.bmm(x, w.permute(0, 2, 1))
    if noise is not None:
        x = x.add_(noise)
    return x


@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        padding=0,  # Padding with respect to the upsampled image.
        resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        demodulate=True,  # Apply weight demodulation?
        flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
            self,
            in_features,  # Number of input features.
            out_features,  # Number of output features.
            bias=True,  # Apply additive bias before the activation function?
            activation='linear',  # Activation function: 'relu', 'lrelu', etc.
            device='cuda',
            lr_multiplier=1,  # Learning rate multiplier.
            bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features], device=device) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init), device=device)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            kernel_size,  # Width and height of the convolution kernel.
            device='cuda',
            bias=True,  # Apply additive bias before the activation function?
            activation='linear',  # Activation function: 'relu', 'lrelu', etc.
            up=1,  # Integer upsampling factor.
            down=1,  # Integer downsampling factor.
            resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
            channels_last=False,  # Expect the input to have memory_format=channels_last?
            trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size], device=device).to(memory_format=memory_format)
        bias = torch.zeros([out_channels], device=device) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join(
            [
                f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
                f'up={self.up}, down={self.down}'])


# ----------------------------------------------------------------------------
@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(
            self,
            z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
            c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
            w_dim,  # Intermediate latent (W) dimensionality.
            num_ws,  # Number of intermediate latents to output, None = do not broadcast.
            num_layers=8,  # Number of mapping layers.
            embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
            layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            device='cuda',
            lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
            w_avg_beta=0.998,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features, device=device)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier, device=device)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def update_w_avg(self, device='device'):
        z = torch.randn([100000, self.z_dim], device=device)
        ws = self.forward(z, None)
        avg_ws = torch.mean(ws, dim=0)[0]
        self.w_avg = self.w_avg * 0.0 + avg_ws

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):

                assert self.w_avg_beta is not None
                assert truncation_cutoff is None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'


# ----------------------------------------------------------------------------
@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels, 0 = first block.
            tmp_channels,  # Number of intermediate channels.
            out_channels,  # Number of output channels.
            resolution,  # Resolution of this block.
            img_channels,  # Number of input color channels.
            first_layer_idx,  # Index of the first layer.
            device='cuda',
            architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            use_fp16=False,  # Use FP16 for this block?
            fp16_channels_last=False,  # Use channels-last memory format with FP16?
            freeze_layers=0,  # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(
                img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last, device=device)

        self.conv0 = Conv2dLayer(
            tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last, device=device)

        self.conv1 = Conv2dLayer(
            tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp,
            channels_last=self.channels_last, device=device)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(
                tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last, device=device)

    def forward(self, x, img, alpha=1.0, first_layer=False, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        # dtype = torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels.
            cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
            resolution,  # Resolution of this block.
            img_channels,  # Number of input color channels.
            device='cuda',
            architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
            mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation, device=device)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation,
            conv_clamp=conv_clamp, device=device)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation, device=device)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim, device=device)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)
        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
            self,
            c_dim,  # Conditioning label (C) dimensionality.
            img_resolution,  # Input resolution.
            img_channels,  # Number of input color channels.
            architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
            channel_base=32768,  # Overall multiplier for the number of channels.
            channel_max=512,  # Maximum number of channels in any layer.
            num_fp16_res=4,  ############## Use FP16 for the N highest resolutions.
            conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
            add_camera_cond=False,  # whether add camera for conditioning
            data_camera_mode='',  #
            device='cuda',
            block_kwargs={},  # Arguments for DiscriminatorBlock.
            mapping_kwargs={},  # Arguments for MappingNetwork.
            epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()

        self.data_camera_mode = data_camera_mode
        self.conditional_dim = c_dim
        self.c_dim = c_dim
        self.add_camera_cond = add_camera_cond
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))

        self.img_channels_drgb = img_channels
        self.img_channels_dmask = 1
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        # We will compute the positional encoding for the camera
        if self.add_camera_cond:
            if self.data_camera_mode == 'shapenet_car' \
                    or self.data_camera_mode == 'shapenet_chair' or self.data_camera_mode == 'shapenet_motorbike' \
                    or self.data_camera_mode == 'renderpeople' or self.data_camera_mode == 'ts_house' \
                    or self.data_camera_mode == 'ts_animal':
                self.camera_dim = 2
                self.camera_dim_enc = 2 + 2 * 2 * 3

            else:
                raise NotImplementedError
            self.c_dim = self.camera_dim_enc + self.c_dim

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if self.c_dim == 0:
            cmap_dim = 0

        # Step 1: set up discriminator for RGB image
        common_kwargs = dict(img_channels=self.img_channels_drgb, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(
                in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, device=device, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if self.c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=self.c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, device=device, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, device=device, **epilogue_kwargs, **common_kwargs)

        # Step 2: set up discriminator for mask image
        common_kwargs = dict(img_channels=self.img_channels_dmask, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        mask_channel_base = channel_base
        mask_channel_max = channel_max
        mask_channels_dict = {res: min(mask_channel_base // res, mask_channel_max) for res in self.block_resolutions + [4]}
        for res in self.block_resolutions:
            in_channels = mask_channels_dict[res] if res < img_resolution else 0
            tmp_channels = mask_channels_dict[res]
            out_channels = mask_channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(
                in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, device=device, **block_kwargs,
                **common_kwargs)
            setattr(self, f'mask_b{res}', block)
            cur_layer_idx += block.num_layers
        if self.c_dim > 0:
            self.mask_mapping = MappingNetwork(
                z_dim=0, c_dim=self.c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, device=device,
                **mapping_kwargs)
        self.mask_b4 = DiscriminatorEpilogue(
            mask_channels_dict[4], cmap_dim=cmap_dim, resolution=4, device=device, **epilogue_kwargs,
            **common_kwargs)

    def pos_enc_angle(self, camera_angle):
        # Encode the camera angles into cos/sin
        if self.data_camera_mode == 'shapenet_car' \
                or self.data_camera_mode == 'shapenet_chair' or self.data_camera_mode == 'shapenet_motorbike' \
                or self.data_camera_mode == 'renderpeople' or self.data_camera_mode == 'ts_house' \
                or self.data_camera_mode == 'ts_animal':
            L = 3
            p_transformed = torch.cat(
                [torch.cat(
                    [torch.sin((2 ** i) * camera_angle),
                     torch.cos((2 ** i) * camera_angle)],
                    dim=-1) for i in range(L)], dim=-1)
            p_transformed = torch.cat([p_transformed, camera_angle], dim=-1)
        else:
            raise NotImplementedError
        return p_transformed

    def forward(self, img, c, update_emas=False, alpha=1.0, mask_pyramid=None, **block_kwargs):
        # Encoding camera condition first
        if self.add_camera_cond:
            if self.conditional_dim == 0:
                c = self.pos_enc_angle(c)
            else:
                condition_c = c[:, :self.conditional_dim]
                pos_encode_c = self.pos_enc_angle(c[:, self.conditional_dim:])
                c = torch.cat([condition_c, pos_encode_c], dim=-1)
        else:
            c = None

        # Step 1: feed the mask image into the discriminator
        _ = update_emas
        mask_x = None
        img_res = img.shape[-1]
        mask_img = img[:, self.img_channels_drgb:self.img_channels_drgb + self.img_channels_dmask]  # This is only supervising the geometry
        for res in self.block_resolutions:
            block = getattr(self, f'mask_b{res}')
            mask_x, mask_img = block(mask_x, mask_img, alpha, (img_res // 2) == res, **block_kwargs)
        mask_cmap = None
        if self.c_dim > 0:
            mask_cmap = self.mask_mapping(None, c)
        mask_x = self.mask_b4(mask_x, mask_img, mask_cmap)

        # Step 2: feed the RGB image into another discriminator
        img_for_tex = img[:, :self.img_channels_drgb, :, :]
        _ = update_emas
        x = None
        img_res = img_for_tex.shape[-1]
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img_for_tex = block(x, img_for_tex, alpha, (img_res // 2) == res, **block_kwargs)
        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img_for_tex, cmap)
        return x, mask_x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
