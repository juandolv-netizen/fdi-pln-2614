# FDI PLN 2614 - Práctica 5: Generación Causal y NER

## Información del Proyecto
- **Integrante**: Juan Diego Olvera Medina
- **Versión**: 1.0

## Descripción

Implementación de un modelo de lenguaje basado en Transformer para dos tareas:

1. **Generación Causal**: Modelo de language modeling que genera texto continuo a partir de un prompt
2. **Clasificación de Entidades Nombradas (NER)**: Fine-tuning del modelo causal para identificar entidades (PER, LOC)

## Instalación

### Desde el wheel (recomendado)

```bash
pip install fdi_pln_2614_p5-1.0-py3-none-any.whl
fdi-pln-2614-p5 --help
```

### Desde el código fuente

```bash
git clone https://github.com/juandolv-netizen/fdi-pln-2614.git
cd fdi-pln-2614
uv install
```

## Uso

### Entrenar el modelo causal

```bash
uv run fdi-pln-2614-p5 entrenar --tarea causal
```

### Entrenar NER

```bash
uv run fdi-pln-2614-p5 entrenar --tarea ner [--pesos path/a/pesos_causal.pth]
```

### Generar texto

```bash
uv run fdi-pln-2614-p5 generar "alice and" [--pesos path/a/pesos_causal.pth] [--max-tokens 100] [--temperature 0.8]
```

### Clasificar entidades nombradas

```bash
uv run fdi-pln-2614-p5 ner archivo.txt [--pesos path/a/pesos_ner.pth]
```

## Estructura del Proyecto

```
fdi-pln-2614/
├── fdi_pln_2614_p5/          # Paquete principal
│   ├── __init__.py
│   ├── main.py               # CLI principal (typer)
│   ├── casual_train.py       # Entrenamiento causal
│   ├── ner_train.py          # Entrenamiento NER
│   ├── llm.py                # Modelo de lenguaje
│   ├── transformer.py        # Bloques transformer
│   ├── attention.py          # Mecanismo de atención
│   ├── tokenizer.py          # Tokenizador BPE
│   ├── corpus.py             # Carga de corpus
│   ├── preprocess.py         # Preprocesamiento
│   ├── p5_causal_2614.pth    # Pesos preentrenados (causal)
│   ├── p5_ner_2614.pth       # Pesos preentrenados (NER)
│   ├── tokenizer.pkl         # Tokenizador entrenado
│   ├── resources/            # Corpus de entrenamiento
│   └── resources_clean/      # Corpus preprocesado
├── pyproject.toml            # Configuración del proyecto
├── README.md                 # Este archivo
└── .gitignore
```

## Arquitectura

### Modelo de Lenguaje (LM)

- **Embedding**: Embedding de tokens + posicional
- **Transformer Blocks**: 6 capas de transformers
  - Multi-Head Attention (4 heads)
  - Feed-Forward Network
  - Layer Normalization
  - Dropout (0.15)
- **Output Head**: Proyección a vocabulario

**Hiperparámetros**:
- Vocab size: 500
- Model dim (d_model): 128
- Attention heads: 4
- Layers: 6
- Context size: 256
- Expansion (MLP): 4

### Tokenizador

BPE (Byte Pair Encoding) con vocab_size=1000

### Esquema NER

Tags: `O`, `B-PER`, `I-PER`, `B-LOC`, `I-LOC`

## Requisitos

- Python >= 3.12
- PyTorch 2.2.2 (CUDA 11.8)
- typer, loguru, rich, dynaconf, python-dotenv, numpy

## Rendimiento

### Modelo Causal
- Loss de validación: ~1.2-1.5 después del entrenamiento
- Capacidad de generar secuencias coherentes

### NER
- Detección correcta de entidades en textos de Lewis Carroll
- Fine-tuning con frozen layers para eficiencia

## Exploración de Hiperparámetros

Se realizaron 3 experimentos registrados en `fdi_pln_2614_p5/experimentos.jsonl`. El corpus de entrenamiento es la concatenación preprocesada de *Alice in Wonderland* y *Through the Looking-Glass*.

| Exp | vocab\_size | n\_layers | dropout | batch\_size | epochs | train\_loss | val\_loss |
|-----|-------------|-----------|---------|-------------|--------|-------------|-----------|
| 1   | 500         | 4         | 0.25    | 64          | 5      | 2.032       | 1.995     |
| 2   | 500         | 4         | 0.25    | 128         | 10     | 1.471       | 2.686     |
| 3   | 2000        | 4         | 0.35    | 128         | 7      | 2.221       | 2.061     |

### Observaciones

- **Exp 1 → Exp 2** (más épocas, mayor batch): el entrenamiento mejoró (train 2.03 → 1.47) pero la validación empeoró significativamente (1.99 → 2.69), señal clara de **sobreajuste**. Con un corpus pequeño (~200 K tokens), 10 épocas son excesivas sin regularización adicional.
- **Exp 2 → Exp 3** (vocab más grande, más dropout): aumentar el vocabulario de 500 a 2 000 tokens perjudicó el rendimiento, ya que el corpus es demasiado pequeño para aprender representaciones útiles de 2 000 tokens. Sin embargo, aumentar el dropout (0.25 → 0.35) mejoró la generalización frente al Exp 2.
- El modelo final (`p5_causal_2614.pth`) usa la configuración de `ModelConfig` con `n_layers=6`, obtenida mediante entrenamiento continuo a partir de los mejores pesos del Exp 1.

### Posibles mejoras

- **Learning rate scheduling**: usar cosine annealing o warmup reduciría el sobreajuste observado en Exp 2 sin necesidad de bajar épocas.
- **Corpus más grande**: el tamaño (~200 K tokens) limita cuánto vocabulario y estructura puede aprender el modelo; añadir más texto de Carroll u obras de época similar mejoraría la cobertura.
- **Weight decay explícito**: `AdamW` con `weight_decay > 0` (p. ej. 0.01) actuaría como regularizador adicional al dropout.
- **Vocab óptimo**: los experimentos sugieren que con este corpus el rango 500–800 tokens es más adecuado que 2 000; un barrido más fino lo confirmaría.

## Nota Técnica

Los archivos `.pth` deben estar en el mismo directorio del módulo o especificarse con `--pesos`.

