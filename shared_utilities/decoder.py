import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

        #  Corrected Conv1D for Downsampling (stride=2 for 72 → 36)
        self.conv_downsample = nn.Conv1d(
            in_channels=layers[0].conv1.in_channels,  
            out_channels=layers[0].conv1.in_channels,  
            kernel_size=3,
            stride=2,  # Ensures 72 → 36
            padding=1
        )

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Forward pass of the decoder with proper cross-attention downsampling.
        """
        # Ensure `cross` (encoder output) is downsampled if needed
        if x.shape[1] > cross.shape[1]:  # If decoder input is longer than encoder output
            cross = self.conv_downsample(cross.permute(0, 2, 1)).permute(0, 2, 1)  # Apply Conv1D & reshape

        # Ensure `cross` matches decoder input length (`x.shape[1]`)
        if cross.shape[1] > x.shape[1]:
            cross = cross[:, -x.shape[1]:, :]  # Trim extra time steps
        elif cross.shape[1] < x.shape[1]:  
            pad = torch.zeros((cross.shape[0], x.shape[1] - cross.shape[1], cross.shape[2]), device=cross.device)
            cross = torch.cat([pad, cross], dim=1)  # Pad at the start

        # Process each decoder layer
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x