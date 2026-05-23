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
git clone https://github.com/tu-usuario/fdi-pln-2614.git
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

## Nota Técnica

Los archivos `.pth` deben estar en el mismo directorio del módulo o especificarse con `--pesos`.

