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

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, expansion, dropout):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, expansion, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.tok_emb.weight = self.lm_head.weight
        
        # Inicialización de pesos estandarizada
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, causal=True):
        B, T = idx.shape
        
        if T > self.max_seq_len:
            raise ValueError(f"Secuencia ({T}) excede límite posicional ({self.max_seq_len})")
            
        # device=idx.device: Crítico. Asegura que el tensor posicional se instancie 
        # directamente en la GPU si la entrada ya está allí.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        
        for block in self.blocks:
            x = block(x, causal=causal)
            
        return self.lm_head(self.ln_f(x))