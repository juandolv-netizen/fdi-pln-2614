# Entrenamiento de LLM causal en base a un corpus
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import time
import json
import dataclasses
from datetime import datetime
from dataclasses import dataclass

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    vocab_size: int = 500
    context_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    expansion: int = 4
    dropout: float = 0.25
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-4
    train_ratio: float = 0.9


def registrar_experimento(config, train_loss, val_loss, tiempo, filepath="experimentos.jsonl"):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "config": dataclasses.asdict(config),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "tiempo_s": tiempo
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro) + "\n")


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

    best_val_loss = float('inf') # Añadido

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        
        # solo si hay mejora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "modelo_preentrenado.pth")
            logger.info(f"Punto de control guardado (val_loss: {best_val_loss:.4f})")

        elapsed = time.time() - t0
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")
    return train_loss, val_loss, elapsed


if __name__ == "__main__":
    import sys
    from llm import LM
    from tokenizer import BPETokenizer
    
    try:
        from corpus import load_corpus
    except ImportError:
        def load_corpus(path):
            from pathlib import Path
            return "\n".join(open(p, encoding="utf-8").read() for p in Path(path).glob("*.txt"))

    corpus = sys.argv[1] if len(sys.argv) > 1 else "resources"
    text = load_corpus(corpus)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ModelConfig()

    tokenizer = BPETokenizer(text, vocab_size=config.vocab_size)
    tokens = tokenizer.encode(text)

    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.context_size,
        expansion=config.expansion,
        dropout=config.dropout,
    ).to(device)

    train_loss, val_loss, elapsed = train(
        model, 
        tokens, 
        epochs=config.epochs, 
        context_size=config.context_size,
        batch_size=config.batch_size,
        lr=config.lr,
        train_ratio=config.train_ratio
    )

    registrar_experimento(config, train_loss, val_loss, elapsed)

    import pickle
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    logger.info("Modelo y tokenizador guardados exitosamente.")

    prompt = "alice and the cat were studying for the exam. what "
    prompt_ids = tokenizer.encode(prompt)
    pred_ids = model.generate(prompt_ids, max_tokens=200)
    logger.opt(colors=True).info(f"<cyan>{prompt}</cyan>{tokenizer.decode(pred_ids)[:500]}")