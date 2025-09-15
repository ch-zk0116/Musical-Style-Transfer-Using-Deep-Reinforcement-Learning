# TransformerVAE.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerVAE(nn.Module):
    """
    A Transformer-based Variational Autoencoder for symbolic music generation.

    This model can be initialized in two ways:
    1. With a config object: `model = TransformerVAE(config)`
       (Used in your training scripts)
    2. With default parameters: `model = TransformerVAE()`
       (Useful for quick instantiation, testing, or inference)
    """
    # [NEW] Nested class to hold default hyperparameters.
    # Using a class with a leading underscore indicates it's for internal use.
    class _DefaultConfig:
        PITCHES = 88
        CHANNELS = 3
        TIMESTEPS = 128
        EMBED_DIM = 512
        LATENT_DIM = 256
        NHEAD = 8
        NUM_ENCODER_LAYERS = 6
        NUM_DECODER_LAYERS = 6
        DROPOUT = 0.1

    def __init__(self, config=None):
        super().__init__()
        
        # [MODIFIED] If no config is provided, use the internal default config.
        if config is None:
            print("Warning: No config object provided. Initializing TransformerVAE with default parameters.")
            config = self._DefaultConfig()

        self.config = config
        self.input_dim = config.PITCHES * config.CHANNELS

        # --- ENCODER ---
        self.fc_in = nn.Linear(self.input_dim, config.EMBED_DIM)
        self.pos_encoder = PositionalEncoding(config.EMBED_DIM, config.DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.EMBED_DIM, nhead=config.NHEAD, dropout=config.DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_ENCODER_LAYERS)
        
        self.fc_mu = nn.Linear(config.EMBED_DIM, config.LATENT_DIM)
        self.fc_logvar = nn.Linear(config.EMBED_DIM, config.LATENT_DIM)

        # --- DECODER ---
        self.fc_latent = nn.Linear(config.LATENT_DIM, config.EMBED_DIM)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.EMBED_DIM, nhead=config.NHEAD, dropout=config.DROPOUT, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.NUM_DECODER_LAYERS)
        self.fc_out = nn.Linear(config.EMBED_DIM, self.input_dim)
        
    def encode(self, x):
        x = x.permute(0, 3, 2, 1).reshape(x.size(0), self.config.TIMESTEPS, -1)
        x = self.fc_in(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        encoded_mean = encoded.mean(dim=1)
        mu = self.fc_mu(encoded_mean)
        logvar = self.fc_logvar(encoded_mean)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        latent_proj = self.fc_latent(z)
        tgt = latent_proj.unsqueeze(1).repeat(1, self.config.TIMESTEPS, 1)
        tgt = self.pos_encoder(tgt)
        memory = tgt
        decoded = self.transformer_decoder(tgt, memory)
        output = self.fc_out(decoded)
        output = output.reshape(output.size(0), self.config.TIMESTEPS, self.config.PITCHES, self.config.CHANNELS)
        output = output.permute(0, 3, 2, 1)
        return torch.sigmoid(output)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar