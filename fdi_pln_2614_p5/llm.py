import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, softmax
from .transformer import TransformerBlock


def n_layers_from_state(state_dict: dict) -> int:
    """Infiere el número de bloques transformer de un state_dict guardado."""
    return len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})


class LM(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, n_layers, max_seq_len, expansion, dropout
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, max_seq_len, expansion, dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, causal=True):
        B, T = idx.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Secuencia ({T}) excede límite posicional ({self.max_seq_len})"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x, causal=causal)

        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            return logits, None

        predicted = logits.flatten(0, 1)
        loss = cross_entropy(predicted, targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_tokens=200, temperature=0.8):
        self.eval()

        ventana = torch.tensor(
            [prompt[-self.max_seq_len :]],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        generados = []
        for _ in range(max_tokens):
            logits, _ = self(ventana)
            next_token_logits = logits[:, -1, :]
            next_token_probs = softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, 1)

            generados.append(next_token_id.item())
            ventana = torch.cat([ventana, next_token_id], dim=1)[:, -self.max_seq_len :]

        return generados
