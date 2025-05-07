import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "../../"))

sys.path.append(os.path.join(PROJECT_ROOT, "shared_utilities"))
from masking import *
from encoder import *
from decoder import *
from attn import *
from embed import *

class BGFormer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, embed='timef', freq='t', activation='gelu', 
                output_attention=False, distil=False, mix=False,
                device=torch.device('mps')):
        super(BGFormer, self).__init__()
        self.pred_len = out_len
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Microscale Attention for Encoder, Decoder, and Cross-Attention
        EncAttn = MicroscaleAttention(d_model, n_heads=n_heads, window_size=12, overlap=6, dropout=dropout)
        DecAttn = DynamicMicroscaleAttention(d_model, n_heads=n_heads, window_size=6, overlap=3, dropout=dropout)
        CrossAttn = DynamicMicroscaleAttention(d_model, n_heads=n_heads, window_size=6, overlap=3, dropout=dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(EncAttn, d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(DecAttn, d_model, n_heads, mix=mix),  # Decoder Self-Attention
                    AttentionLayer(CrossAttn, d_model, n_heads, mix=False),  # Cross-Attention
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Output Projection
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass of GPFormerLite.
        """
        # Encoder forward pass
        enc_out = self.enc_embedding(x_enc)
        # print(f"DEBUG: embed_out.shape_before_encoder = {enc_out.shape}")
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(f"DEBUG: enc_out.shape = {enc_out.shape}")

        # Decoder forward pass
        dec_out = self.dec_embedding(x_dec)
        # print(f"DEBUG: embedding_dec_out.shape = {dec_out.shape}")
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # print(f"DEBUG: decoder_dec_out.shape = {dec_out.shape}")
        dec_out = self.projection(dec_out)
        # print(f"DEBUG: final_dec_out.shape = {dec_out.shape}")

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

