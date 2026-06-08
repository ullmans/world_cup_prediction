import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class WorldCupTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(WorldCupTransformer, self).__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        self.softplus = nn.Softplus()

    def forward(self, team_a_seq, team_b_seq):
        a_embedded = self.input_projection(team_a_seq)
        a_encoded = self.pos_encoder(a_embedded)
        a_out = self.transformer_encoder(a_encoded)
        a_final_state = a_out[:, -1, :]

        b_embedded = self.input_projection(team_b_seq)
        b_encoded = self.pos_encoder(b_embedded)
        b_out = self.transformer_encoder(b_encoded)
        b_final_state = b_out[:, -1, :]

        combined_state = torch.cat((a_final_state, b_final_state), dim=1)
        raw_lambdas = self.fc_out(combined_state)
        predicted_lambdas = self.softplus(raw_lambdas)

        return predicted_lambdas
