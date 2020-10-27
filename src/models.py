"""
PyTorch implementation of 
A FULLY CONVOLUTIONAL NEURAL NETWORK FOR SPEECH ENHANCEMENT paper
https://arxiv.org/pdf/1609.07132.pdf
"""
import torch
from torch import nn
import pytorch_tools as pt

class CRCED(torch.nn.modules.Module):
    """
    Cascaded Redundant Convolutional Encoder-Decoder Network (CR-CED)
    """
    def __init__(
        self,
        in_channels=1,
        n_fft=80,
        n_segments=8,
        channels=[18, 30, 8],
        filters=[9, 5, 9],
        n_layers=5,
        norm_name='abn',
        norm_act='relu',
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.n_segments = n_segments
        self.norm_act = norm_act
        self.norm_layer = pt.modules.bn_from_name(norm_name)
        self.n_layers = n_layers
        self.in_channels = in_channels
        
        self.blocks = []
        self.first = True
        for _ in range(n_layers):
            self.blocks.append(self._make_block(channels, filters))
            
        self.features = nn.Sequential(*self.blocks)
    
        self.last_conv = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=(1, 1),
            padding=0,
        )

        self._initialize_weights()

    def forward(self, x):
        """
        Args:
            x: Batch of melspectograms with shape (N, 3, N_FFT, N_SEGMENTS)
        Returns:
            denoised segment
        """
        x = self.features(x)
        x = self.last_conv(x)

        return x
    
    def _make_block(self, channels, filters):
        layers = []
        for c, f in zip(channels, filters):
            # Shrink along time axis
            if self.first:
                conv2d = nn.Conv2d(self.in_channels, c, kernel_size=(f, self.n_segments), padding=(f // 2, 0))
                self.first = False
            else:
                conv2d = nn.Conv2d(self.in_channels, c, kernel_size=(f, 1), padding=(f // 2, 0))
            layers += [conv2d, self.norm_layer(c, activation=self.norm_act)]
            self.in_channels = c
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)