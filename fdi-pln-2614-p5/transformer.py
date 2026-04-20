import torch
import torch.nn as nn
from attention import Attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, expansion, dropout):
        super().__init__()
        # bias=False en LayerNorm ahorra memoria y cómputo sin impacto en rendimiento
        self.ln_1 = nn.LayerNorm(d_model, bias=False)
        self.attn = Attention(d_model, n_heads, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model, bias=False)
        
        # Factor de expansión dinámico en lugar de constante fija
        hidden_dim = int(d_model * expansion)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, causal=True):
        # Evitar asignaciones intermedias (ej. y = self.attn(...)) reduce retención
        # de tensores en memoria durante la creación del grafo para Autograd
        x = x + self.attn(self.ln_1(x), causal=causal)
        x = x + self.mlp(self.ln_2(x))
        return x