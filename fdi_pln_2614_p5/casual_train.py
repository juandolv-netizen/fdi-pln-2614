# casual_train.py
import os
import time
import json
import pickle
import dataclasses
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    vocab_size: int = 500
    context_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    expansion: int = 4
    dropout: float = 0.15
    batch_size: int = 128
    epochs: int = 7
    lr: float = 3e-4
    train_ratio: float = 0.9
    val_freq: int = 5


def registrar_experimento(
    config, train_loss, val_loss, tiempo, filepath="experimentos.jsonl"
):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "config": dataclasses.asdict(config),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "tiempo_s": tiempo,
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
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        ),
    )


def _run_epoch(model, dataloader, optimizer=None):
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    scaler = (
        torch.cuda.amp.GradScaler() if optimizer and device.type == "cuda" else None
    )

    if optimizer:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if optimizer:
            optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            _, loss = model(x, y)

        if optimizer:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

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
    val_freq=5,
):
    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    val_loss = None

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)

        if (epoch + 1) % val_freq == 0 or epoch == epochs - 1:
            val_loss = _run_epoch(model, val_dl, None)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "p5_causal_2614.pth")
                logger.info(
                    f"Punto de control guardado (val_loss: {best_val_loss:.4f})"
                )

        elapsed = time.time() - t0
        val_log = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_log} | tiempo={elapsed:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")
    return train_loss, val_loss, elapsed


def train_causal():
    """Entrena el modelo de generación causal (language modeling)."""
    import sys
    from .llm import LM
    from .tokenizer import BPETokenizer
    from .preprocess import run_preprocessing

    # Get module directory for paths
    module_dir = Path(__file__).parent

    # 0. Preprocesamiento dinámico
    input_corpus_dir = module_dir / "resources"
    processed_dir = module_dir / "resources_clean"

    logger.info(
        f"Preprocesando corpus desde '{input_corpus_dir}' hacia '{processed_dir}'..."
    )
    run_preprocessing(
        input_dir=str(input_corpus_dir),
        output_dir=str(processed_dir),
        exclude_files={"Natural_Language_Processing_with_Python.txt"},
    )

    # 1. Carga de corpus
    try:
        from .corpus import load_corpus

        text = load_corpus(str(processed_dir))
    except ImportError:

        def load_corpus(path):
            p = Path(path)
            return "\n".join(
                open(f, encoding="utf-8", errors="ignore").read()
                for f in p.glob("*.txt")
            )

        text = load_corpus(str(processed_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig()

    # 2. Carga o instanciación del tokenizador
    tokenizer_path = module_dir / "tokenizer.pkl"
    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        logger.info("Tokenizador cargado desde disco.")
    else:
        tokenizer = BPETokenizer(text, vocab_size=config.vocab_size)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        logger.info("Nuevo tokenizador creado y guardado.")

    tokens = tokenizer.encode(text)

    # 3. Instanciación del modelo
    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.context_size,
        expansion=config.expansion,
        dropout=config.dropout,
    ).to(device)

    # 4. Inyección de pesos para entrenamiento continuo
    pretrained_path = module_dir / "modelo_preentrenado.pth"
    if pretrained_path.exists():
        model.load_state_dict(
            torch.load(pretrained_path, map_location=device, weights_only=True)
        )
        logger.info("Pesos del modelo cargados. Iniciando entrenamiento continuo.")

    # 5. Compilación segura del modelo
    if hasattr(torch, "compile") and device == "cuda":
        try:
            model = torch.compile(model)
        except RuntimeError:
            logger.warning(
                "torch.compile no soportado en Python 3.12+. Omitiendo compilación."
            )

    # 6. Ejecución del entrenamiento
    train_loss, val_loss, elapsed = train(
        model,
        tokens,
        epochs=config.epochs,
        context_size=config.context_size,
        batch_size=config.batch_size,
        lr=config.lr,
        train_ratio=config.train_ratio,
        val_freq=config.val_freq,
    )

    registrar_experimento(
        config,
        train_loss,
        val_loss,
        elapsed,
        filepath=str(module_dir / "experimentos.jsonl"),
    )

    prompt = "alice and the cat were studying for the exam. what "
    prompt_ids = tokenizer.encode(prompt)
    pred_ids = model.generate(prompt_ids, max_tokens=200)
    logger.opt(colors=True).info(
        f"<cyan>{prompt}</cyan>{tokenizer.decode(pred_ids)[:500]}"
    )


if __name__ == "__main__":
    import sys
    from .llm import LM
    from .tokenizer import BPETokenizer
    from .preprocess import run_preprocessing

    # 0. Preprocesamiento dinámico
    input_corpus_dir = sys.argv[1] if len(sys.argv) > 1 else "resources"
    processed_dir = f"{input_corpus_dir}_clean"

    logger.info(
        f"Preprocesando corpus desde '{input_corpus_dir}' hacia '{processed_dir}'..."
    )
    run_preprocessing(
        input_dir=input_corpus_dir,
        output_dir=processed_dir,
        exclude_files={"Natural_Language_Processing_with_Python.txt"},
    )

    # 1. Carga de corpus
    try:
        from .corpus import load_corpus

        text = load_corpus(processed_dir)
    except ImportError:

        def load_corpus(path):
            p = Path(path)
            return "\n".join(
                open(f, encoding="utf-8", errors="ignore").read()
                for f in p.glob("*.txt")
            )

        text = load_corpus(processed_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig()

    # 2. Carga o instanciación del tokenizador
    if os.path.exists("tokenizer.pkl"):
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        logger.info("Tokenizador cargado desde disco.")
    else:
        tokenizer = BPETokenizer(text, vocab_size=config.vocab_size)
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        logger.info("Nuevo tokenizador creado y guardado.")

    tokens = tokenizer.encode(text)

    # 3. Instanciación del modelo
    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.context_size,
        expansion=config.expansion,
        dropout=config.dropout,
    ).to(device)

    # 4. Inyección de pesos para entrenamiento continuo
    if os.path.exists("modelo_preentrenado.pth"):
        model.load_state_dict(
            torch.load(
                "modelo_preentrenado.pth", map_location=device, weights_only=True
            )
        )
        logger.info("Pesos del modelo cargados. Iniciando entrenamiento continuo.")

    # 5. Compilación segura del modelo
    if hasattr(torch, "compile") and device == "cuda":
        try:
            model = torch.compile(model)
        except RuntimeError:
            logger.warning(
                "torch.compile no soportado en Python 3.12+. Omitiendo compilación."
            )

    # 6. Ejecución del entrenamiento
    train_loss, val_loss, elapsed = train(
        model,
        tokens,
        epochs=config.epochs,
        context_size=config.context_size,
        batch_size=config.batch_size,
        lr=config.lr,
        train_ratio=config.train_ratio,
        val_freq=config.val_freq,
    )

    registrar_experimento(config, train_loss, val_loss, elapsed)

    prompt = "alice and the cat were studying for the exam. what "
    prompt_ids = tokenizer.encode(prompt)
    pred_ids = model.generate(prompt_ids, max_tokens=200)
    logger.opt(colors=True).info(
        f"<cyan>{prompt}</cyan>{tokenizer.decode(pred_ids)[:500]}"
    )
