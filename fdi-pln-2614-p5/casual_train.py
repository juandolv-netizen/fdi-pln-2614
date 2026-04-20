# Entrenamiento de LLM causal en base a un corpus
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import time
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

def _make_dataloaders(tokens, context_size, batch_size, train_ratio=0.9):
    data = torch.tensor(tokens, dtype=torch.long)
    split = int(train_ratio * len(data))
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)
    logger.info(f"Train: {len(train_ds):,} muestras, Val: {len(val_ds):,}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )

def _run_epoch(model, dataloader, optimizer=None):
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    if optimizer:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        if optimizer:
            optimizer.zero_grad()

        _, loss = model(x, y)

        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n

def train(
    model,
    tokens,
    epochs=5,
    context_size=128,
    batch_size=64,
    lr=3e-4,
    train_ratio=0.9,
):
    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        elapsed = time.time() - t0
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")

if __name__ == "__main__":
    import sys
    # Importaciones corregidas según archivos proporcionados
    from llm import LM
    from tokenizer import BPETokenizer
    
    # Se asume que load_corpus está disponible en el entorno o en un archivo corpus.py local
    try:
        from corpus import load_corpus
    except ImportError:
        def load_corpus(path):
            from pathlib import Path
            return "\n".join(open(p, encoding="utf-8").read() for p in Path(path).glob("*.txt"))

    corpus = sys.argv[1] if len(sys.argv) > 1 else "resources"
    text = load_corpus(corpus)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = 300
    CONTEXT_SIZE = 128

    tokenizer = BPETokenizer(text, vocab_size=VOCAB_SIZE)
    tokens = tokenizer.encode(text)

    # Instanciación de LM con los argumentos esperados por llm.py
    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_seq_len=CONTEXT_SIZE,
        expansion=4,
        dropout=0.1,
    ).to(device)

    train(model, tokens, epochs=5, context_size=CONTEXT_SIZE)

    prompt = "alice and the cat were studying for the exam. what "
    # Codificación del prompt y generación
    prompt_ids = tokenizer.encode(prompt)
    pred_ids = model.generate(prompt_ids, max_tokens=200)
    logger.opt(colors=True).info(f"<cyan>{prompt}</cyan>{tokenizer.decode(pred_ids)[:500]}")