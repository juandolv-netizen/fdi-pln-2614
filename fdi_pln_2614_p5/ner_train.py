import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from loguru import logger
import pickle
from pathlib import Path

from .llm import LM
from .casual_train import ModelConfig

# Vocabulario de etiquetas estático
TAG2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}


class NerDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=256):
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["words"]
        labels = item["labels"]

        word_ids = []
        label_ids = []

        for word, label in zip(words, labels):
            # Codificamos la palabra añadiendo un espacio para simular el flujo del texto
            sub_tokens = self.tokenizer.encode(word + " ")
            if not sub_tokens:
                continue

            word_ids.extend(sub_tokens)
            # El primer sub-token recibe la etiqueta real
            label_ids.append(TAG2ID[label])
            # Los sub-tokens restantes se ignoran en el cálculo de pérdida (-100)
            label_ids.extend([-100] * (len(sub_tokens) - 1))
            
            # Truncate if exceeding max_seq_len
            if len(word_ids) >= self.max_seq_len:
                word_ids = word_ids[:self.max_seq_len]
                label_ids = label_ids[:self.max_seq_len]
                break

        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(
            label_ids, dtype=torch.long
        )


def collate_fn(batch):
    inputs, targets = zip(*batch)
    # Rellenar secuencias cortas con 0 (inputs) y -100 (targets)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputs_padded, targets_padded


def train_ner():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ModelConfig()

    # Get the path to the tokenizer relative to this module
    module_dir = Path(__file__).parent
    tokenizer_path = module_dir / "tokenizer.pkl"
    
    # Handle pickle import issue by fixing sys.modules
    import sys
    if 'tokenizer' not in sys.modules:
        from . import tokenizer as tokenizer_module
        sys.modules['tokenizer'] = tokenizer_module
    
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        logger.error(f"Error loading tokenizer from {tokenizer_path}. Regenerating...")
        # If tokenizer can't be loaded, we need to regenerate it
        # This would require training data, so for now just raise
        raise

    # Inicializar modelo con la misma configuración
    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.context_size,
        expansion=config.expansion,
        dropout=config.dropout,
    )

    # Cargar pesos base
    pretrained_path = module_dir / "p5_causal_2614.pth"
    if pretrained_path.exists():
        try:
            state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
            # Filtrar pesos que no coincidan en tamaño (vocabulary mismatch)
            model_state = model.state_dict()
            compatible_weights = {}
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_weights[key] = value
            
            # Cargar solo los pesos compatibles
            if compatible_weights:
                model.load_state_dict(compatible_weights, strict=False)
                loaded_pct = (len(compatible_weights) / len(model_state)) * 100
                logger.info(f"Pesos causales cargados: {len(compatible_weights)}/{len(model_state)} parámetros ({loaded_pct:.1f}%)")
            else:
                logger.warning("No hay pesos compatibles para cargar")
        except Exception as e:
            logger.error(f"Error cargando pesos base: {e}")
            logger.warning("Continuando sin pesos preentrenados...")
    else:
        logger.warning(f"No se encontraron pesos causales en {pretrained_path}")
        logger.warning("Asegúrate de entrenar el modelo causal primero con: entrenar -t causal")

    # Adaptar para NER: Cambiar cabeza de salida a 5 clases y desvincular pesos (weight tying)
    model.lm_head = nn.Linear(config.d_model, len(TAG2ID), bias=False)

    # Congelar capas base para ahorrar memoria y acelerar el fine-tuning
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False

    model.to(device)

    # Preparar Datos
    ner_dataset_path = module_dir / "resources" / "ner_dataset.json"
    dataset = NerDataset(ner_dataset_path, tokenizer, max_seq_len=config.context_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Optimizador y Función de Pérdida
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    # PyTorch ignorará automáticamente los índices -100
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    logger.info("Iniciando Fine-Tuning para NER...")

    epochs = 3  # Suelen ser pocas épocas para un fine-tuning congelado
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Pasamos targets=None para que llm.py no calcule su cross_entropy interna
            logits, _ = model(inputs, targets=None, causal=True)

            # Reestructurar dimensiones para la función de pérdida
            # logits: (Batch, Seq, 5) -> (Batch * Seq, 5)
            # targets: (Batch, Seq) -> (Batch * Seq)
            loss = criterion(logits.view(-1, len(TAG2ID)), targets.view(-1))

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Época {epoch + 1}/{epochs} | Lote {batch_idx} | Pérdida: {loss.item():.4f}"
                )

    pth_path = module_dir / "p5_ner_2614.pth"
    torch.save(model.state_dict(), pth_path)
    logger.info(f"Entrenamiento finalizado. Pesos guardados en {pth_path}")


if __name__ == "__main__":
    train_ner()
