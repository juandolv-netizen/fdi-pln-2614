import pickle
from pathlib import Path
import torch
import torch.nn as nn
import typer
from loguru import logger
from rich.console import Console

from .casual_train import ModelConfig
from .llm import LM

# Get the path to the module directory for accessing resources
MODULE_DIR = Path(__file__).parent

app = typer.Typer(help="CLI para la Práctica 5 de PLN - 2614")
console = Console()

TAG2ID = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}
ID2TAG = {v: k for k, v in TAG2ID.items()}


@app.command()
def entrenar(
    tarea: str = typer.Option(
        ..., "--tarea", "-t", help="Tarea a entrenar: 'causal' o 'ner'"
    ),
):
    """Entrena la tarea de generación de texto (causal) o el clasificador de entidades (ner)."""
    if tarea == "causal":
        logger.info("Iniciando entrenamiento causal...")
        from .casual_train import train_causal

        train_causal()
    elif tarea == "ner":
        logger.info("Iniciando entrenamiento de NER...")
        from .ner_train import train_ner

        train_ner()
    else:
        console.print(
            f"[bold red]Error:[/bold red] Tarea '{tarea}' no reconocida. Opciones válidas: 'causal', 'ner'."
        )
        raise typer.Exit(code=1)


@app.command()
def generar(
    prompt: str = typer.Argument(..., help="Texto inicial para activar la generación"),
    pesos: Path = typer.Option(
        None,
        "--pesos",
        "-p",
        help="Ruta al archivo de pesos causales (.pth)",
    ),
    max_tokens: int = typer.Option(
        50, "--max-tokens", "-m", help="Cantidad máxima de tokens a generar"
    ),
    temperature: float = typer.Option(
        0.8, "--temperature", "-temp", help="Temperatura de muestreo"
    ),
):
    """Genera texto continuo a partir de un prompt utilizando los pesos causales."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use default path if not provided
    if pesos is None:
        pesos = MODULE_DIR / "p5_causal_2614.pth"
    else:
        pesos = Path(pesos)

    if not pesos.exists():
        logger.error(f"Fichero de pesos no encontrado en: {pesos}")
        raise typer.Exit(code=1)

    tokenizer_path = MODULE_DIR / "tokenizer.pkl"
    if not tokenizer_path.exists():
        logger.error(f"Fichero 'tokenizer.pkl' no encontrado en: {tokenizer_path}")
        raise typer.Exit(code=1)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    config = ModelConfig()
    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.context_size,
        expansion=config.expansion,
        dropout=config.dropout,
    )

    model.load_state_dict(torch.load(pesos, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(prompt)

    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids, max_tokens=max_tokens, temperature=temperature
        )

    resultado = tokenizer.decode(prompt_ids + generated_ids)
    console.print(f"\n[bold green]Resultado generado:[/bold green]\n{resultado}")


@app.command()
def ner(
    fichero: Path = typer.Argument(
        ..., help="Ruta al archivo de texto (.txt) a procesar"
    ),
    pesos: Path = typer.Option(
        None,
        "--pesos",
        "-p",
        help="Ruta al archivo de pesos ajustados para NER (.pth)",
    ),
):
    """Analiza un fichero de texto y extrae la lista de entidades nombradas encontradas."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use default path if not provided
    if pesos is None:
        pesos = MODULE_DIR / "p5_ner_2614.pth"
    else:
        pesos = Path(pesos)

    if not fichero.exists():
        logger.error(f"Fichero de entrada no encontrado en: {fichero}")
        raise typer.Exit(code=1)

    if not pesos.exists():
        logger.error(f"Fichero de pesos NER no encontrado en: {pesos}")
        raise typer.Exit(code=1)

    tokenizer_path = MODULE_DIR / "tokenizer.pkl"
    if not tokenizer_path.exists():
        logger.error(f"Fichero 'tokenizer.pkl' no encontrado en: {tokenizer_path}")
        raise typer.Exit(code=1)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    texto = fichero.read_text(encoding="utf-8")
    palabras = texto.split()

    config = ModelConfig()
    model = LM(
        vocab_size=len(tokenizer.vocab),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        max_seq_len=config.context_size,
        expansion=config.expansion,
        dropout=config.dropout,
    )

    # Reemplazar la cabeza lineal por la de clasificación de tokens NER
    model.lm_head = nn.Linear(config.d_model, len(TAG2ID), bias=False)
    model.load_state_dict(torch.load(pesos, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    word_ids = []
    word_to_subtoken_map = []

    for palabra in palabras:
        sub_tokens = tokenizer.encode(palabra + " ")
        if sub_tokens:
            word_to_subtoken_map.append(len(word_ids))
            word_ids.extend(sub_tokens)

    if len(word_ids) > config.context_size:
        logger.warning(
            f"La secuencia excede la longitud máxima ({config.context_size}). Se truncará."
        )
        word_ids = word_ids[: config.context_size]

    input_tensor = torch.tensor([word_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, _ = model(input_tensor, targets=None, causal=True)
        predicciones = torch.argmax(logits, dim=-1)[0]

    console.print("\n[bold green]Entidades Nombradas Detectadas:[/bold green]")
    entidades_encontradas = False

    for i, idx_subtoken in enumerate(word_to_subtoken_map):
        if idx_subtoken >= len(predicciones):
            break

        id_etiqueta = predicciones[idx_subtoken].item()
        etiqueta = ID2TAG.get(id_etiqueta, "O")

        if etiqueta != "O":
            console.print(
                f"- [bold cyan]{palabras[i]}[/bold cyan] : [yellow]{etiqueta}[/yellow]"
            )
            entidades_encontradas = True

    if not entidades_encontradas:
        console.print("[pálido]No se encontraron entidades en el archivo.[/pálido]")


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    app()
