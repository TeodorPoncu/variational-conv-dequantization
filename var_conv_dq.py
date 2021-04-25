import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch.distributions import Normal


class TestConvDequantize(nn.Module):
    """
    Please refer to VariationalConvDequantize class down bellow for implementation details
    This is a dummy class with extended return types for testing purposes and sanity checks
    """

    def __init__(self, in_channels: int, z_channels: int):
        super(TestConvDequantize, self).__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.latent = nn.Sequential(
            nn.Conv2d(in_channels, z_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Linear(z_channels, 2)
        )

    def dequantize(self, x: torch.Tensor):
        latents = self.latent(x)
        b, c, h, w = latents.shape
        latents = latents.permute(0, 2, 3, 1)
        latents = latents.reshape(b * h * w, self.z_channels)
        latents = self.mlp(latents)
        mu, var = torch.chunk(latents, chunks=2, dim=1)
        noise_original = torch.randn(size=(b * h * w, self.in_channels))
        noise = (noise_original + mu) * var
        noise = noise.view(b, h, w, self.in_channels)
        noise = noise.permute(0, 3, 1, 2)
        return noise, noise_original, mu, var

    def forward(self, x: torch.Tensor):
        variational, noise, mu, var = self.dequantize(x)
        dequantize = x + variational
        return dequantize, variational, noise, mu, var


class VariationalConvDequantize(nn.Module):
    """
    Module that performs variational dequantization similarly to Flow++ (https://arxiv.org/abs/1902.00275)

    Used when dealing with spatially dependent quantized embeddings, i.e mu and var are obtained from a feature
    vector that is the result of a convolution operation with kernel_size > 1

    The feature vector for z_{B x H x W} is obtained by performing a convolution around z_{B x H x W}, then a MLP
    extracts mu_{B x H x W}, respectively var_{B x H x W}
    """
    def __init__(self, in_channels: int, z_channels: int):
        super(VariationalConvDequantize, self).__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels

        # use a convolution as a non-linear for spatial awareness and cheap dimensionality reduction
        # can change kernel size to (1, 1) in order to obtain classic variational dequantization
        self.latent = nn.Sequential(
            nn.Conv2d(in_channels, z_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
        )

        # use a mlp to get mean and variance
        self.mlp = nn.Sequential(
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Linear(z_channels, 2)
        )

        self.gaussian = Normal(loc=0., scale=1., )

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        # reduce dimensionality
        latents = self.latent(x)

        # get latent sizes, only C dimension is different from input
        b, c, h, w = latents.shape
        # a linear takes input of form B x D
        # swap axes to perform computation for each spatial position
        # B, C, H, W -> B * H * W, C
        latents = latents.permute(0, 2, 3, 1)
        latents = latents.reshape(b * h * w, self.z_channels)

        # get mu and var
        latents = self.mlp(latents)
        mu, var = torch.chunk(latents, chunks=2, dim=1)

        # sample gaussian noise and add variational parameters
        noise_original = torch.randn(size=(b * h * w, self.in_channels))
        noise = (noise_original + mu) * var

        # rehsape to original shape
        noise = noise.view(b, h, w, self.in_channels)
        # swap axis to preserve spatial ordering
        noise = noise.permute(0, 3, 1, 2)

        return noise

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # return variational noise
        variational = self.dequantize(x)

        # add gaussian noise
        dequantize = x + variational

        return dequantize, variational


def run_tests():
    module = TestConvDequantize(in_channels=512, z_channels=32)
    y = torch.randn(16, 512, 32, 32)
    dequantized, noise, noise_original, mu, var = module.forward(y)
    x = dequantized[0, :, 0, 0]
    n = noise[0, :, 0, 0]

    assert y.shape == noise.shape, "Failed noise generation, shape mismatch"
    assert torch.allclose((x - n), y[0, :, 0, 0]), "Failed operation check, original input is not equal to self + noise"
    assert torch.allclose(noise_original[0, :], n / var[0] - mu[0]) \
        , "Failed operation order check, variational features do not match " \
          "their original statistics at spatial positions "


if __name__ == '__main__':
    run_tests()
