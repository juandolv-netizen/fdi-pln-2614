# Práctica 1: Agente de Negociación PLN

**Equipo 14**
**Integrantes:** Olvera Medina Juan Diego

---

## Descripción

Agente conversacional autónomo para la Práctica 1 de PLN. Participa en una simulación multi-agente de trueques, comunicándose con otros agentes a través de un servidor central ("Butler") para alcanzar un objetivo asignado mediante negociación en lenguaje natural.

### Arquitectura

```
src/fdi_pln_2614_p1/
├── main.py          # Punto de entrada CLI (click + uvicorn)
├── server.py        # Servidor FastAPI + bucle de correo asíncrono
├── core/
│   └── config.py    # Configuración con dynaconf
├── butler/
│   └── client.py    # Cliente REST del Butler
└── agent/
    ├── classifier.py  # Clasificador de fase de negociación (reglas)
    └── core_agent.py  # Agente pydantic-ai + prompt dinámico + parser
```

### Decisiones técnicas

1. **LLM local (Llama 3.2 vía Ollama):** Conectado con `pydantic-ai` usando un `OpenAIProvider` apuntando al endpoint local de Ollama (`/v1`).
2. **Prompt dinámico:** El system prompt se construye en cada llamada con `@agent.system_prompt`, inyectando el inventario propio, el inventario del oponente (obtenido del Butler), el objetivo actual y la fase de negociación detectada.
3. **Clasificador de fase por reglas:** `classifier.py` detecta ACCEPTANCE / REJECTION / COUNTER_OFFER / OPENING con listas de palabras clave. La detección se hace antes de invocar al LLM para inyectar instrucciones específicas de acción en el prompt.
4. **Cliente Butler separado:** Toda la lógica HTTP está en `butler/client.py`, separada del bucle de control.
5. **Parseo robusto:** `parse_offer()` extrae JSON de la salida del LLM con regex, tolerando alucinaciones de formato habituales en Llama 3.
6. **Anti-duplicados:** El bucle mantiene solo el mensaje más reciente por remitente antes de invocar al LLM.
7. **Reputación de oponentes:** Se mantiene un score por alias (0.0–1.0) que sube con aceptaciones (+0.15) y baja con rechazos (−0.10), visible en el prompt para modular la generosidad de las ofertas.
8. **Proactividad:** Si el buzón está vacío al arrancar, el agente toma la iniciativa y contacta al primer agente disponible.

## Ejecución

```bash
# Instalar dependencias
uv sync

# Arrancar (la URL del Butler se lee de la variable de entorno)
FDI_PLN__BUTLER_ADDRESS=http://<host>:<puerto> uv run fdi-pln-2614-p1

# O usando el puerto por defecto (8001) con el Butler local de settings.toml
uv run fdi-pln-2614-p1
```

## Configuración

Editar `settings.toml` para cambiar el alias del agente o el modelo LLM:

```toml
[LLM]
MODEL_NAME = "llama3.2"
BASE_URL = "http://127.0.0.1:11434/v1"

[AGENT]
ALIAS = "G14"

[PLN]
BUTLER_ADDRESS = "http://127.0.0.1:7719"  # sobreescribible con FDI_PLN__BUTLER_ADDRESS
```

---

## Informe de desarrollo

### Complicaciones encontradas y soluciones

#### 1. Bug del clasificador: falso ACCEPTANCE con "no acepto"

**Problema:** `classify_message("no acepto, es muy caro")` devolvía `MessagePhase.ACCEPTANCE` porque `"acepto"` es subcadena de `"no acepto"`. El agente interpretaba rechazos como aceptaciones y enviaba paquetes sin haber cerrado ningún trato.

**Solución:** Reordenar las comprobaciones: verificar `_REJECTION` antes que `_ACCEPTANCE`. Al encontrar primero "no acepto" en la lista de rechazo, nunca se llega a la búsqueda de "acepto". Adicionalmente se eliminó la palabra `"trato"` sola de la lista de aceptación, ya que "¿hacemos un trato?" generaba falsos positivos; se sustituyó por `"trato hecho"` y `"ok trato"`.


#### 2. Cantidades fraccionarias

**Problema:** Llama 3.2 generaba mensajes con cantidades no enteras ("Te doy 1 madera a cambio de 0,5 piedra"). El agente de Profe tiene un modelo `Acuerdo` con campo `envio_numero: int` que lanza `ValidationError` al recibir `"0.5"`, lo que terminaba en crash de Profe.

**Solución:** Añadir regla explícita al system prompt: `"TODAS las cantidades deben ser números ENTEROS positivos (1, 2, 3...). NUNCA fracciones ni decimales."`.

#### 3. El LLM alucinaba items inexistentes

**Problema:** G14 ofrecía y pedía items que no existen en el inventario de ningún agente ("1 rosa", "2 piedra"), haciendo imposible cerrar tratos. El system prompt ya decía "solo ofrece items que tienes", pero Llama 3.2 lo ignoraba porque no tenía información sobre qué tiene el oponente.

**Solución:** Obtener el inventario del oponente con `GET /info?agente=<alias>` antes de cada llamada al LLM, e incluirlo en el prompt como sección separada con la instrucción "solo pide items que el oponente realmente tiene". El formato cambió de texto plano a lista explícita por ítem (`"- arroz: 2 unidades"`) para que el modelo lo lea con menos ambigüedad.

---

### Nuevos descubrimientos

#### Arquitectura del Butler: modos y endpoints

El Butler tiene dos ejes de configuración independientes que no están documentados en el enunciado y que condicionan completamente el comportamiento:

| Flag | Efecto |
|------|--------|
| `--buzon` | Activa `POST /carta`, `DELETE /mail/{uid}`, campo `Buzon` en `/info` |
| `--monopuesto` | Identifica agentes por `?agente=` en lugar de por IP de cliente |

Sin `--monopuesto`, varios agentes en la misma máquina comparten identidad (misma IP). Sin `--buzon`, no hay mensajería: solo se pueden enviar paquetes directamente.

#### Velocidad de inferencia de Llama 3.2

Llama 3.2 (3B parámetros) tarda entre 1.5 y 3 minutos por respuesta en CPU sin aceleración. Esto hace que el ritmo de negociación sea extremadamente lento. Alternativas exploradas:
- `llama3.2:1b`: ~3x más rápido, peor calidad de instrucciones
- `qwen2.5:3b`: velocidad similar, mejor seguimiento de JSON estructurado
- Modelos 7B–8B: mayor calidad pero más lentos en CPU; recomendables solo con GPU

#### dynaconf y variables de entorno anidadas

La librería `dynaconf` con `nested_separator="__"` permite sobreescribir configuración anidada desde variables de entorno. La variable `FDI_PLN__BUTLER_ADDRESS` mapea a `settings.PLN.BUTLER_ADDRESS`, lo que permite cambiar el Butler sin modificar `settings.toml`, útil para el entorno de clase vs. el local.

---

### Comparativa de modelos LLM para el agente de trueque

> **Nota:** Los tiempos son estimaciones basadas en CPU de gama media sin aceleración GPU (i5/Ryzen 5, ~16 GB RAM). Los resultados reales varían según hardware. La evaluación de calidad es cualitativa, basada en el comportamiento conocido de cada familia de modelos con tareas de JSON estructurado y seguimiento de instrucciones en español.

#### Criterios evaluados

El agente tiene requisitos específicos que no todos los modelos cumplen igual:

1. **JSON estricto:** la salida debe ser siempre `{"msg": "...", "env": null}` sin texto extra
2. **Seguimiento de restricciones:** respetar inventario, cantidades enteras, no inventar items
3. **Español:** calidad del texto generado en los mensajes de negociación
4. **Velocidad en CPU:** tiempo estimado por respuesta (el cuello de botella del agente)
5. **Tamaño descargable:** peso del modelo en disco

#### Tabla comparativa

| Modelo (Ollama) | Parámetros | Tamaño | CPU ~tiempo/resp | JSON fiable | Restricciones | Español | Recomendado para |
|---|---|---|---|---|---|---|---|
| `llama3.2` *(actual)* | 3B | 2.0 GB | 2–3 min | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ | Baseline |
| `llama3.2:1b` | 1B | 1.3 GB | 40–60 s | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | Pruebas rápidas |
| `qwen2.5:3b` | 3B | 2.0 GB | 1–2 min | ★★★★★ | ★★★★☆ | ★★★☆☆ | **Mejor relación velocidad/calidad JSON** |
| `qwen2.5:7b` | 7B | 4.7 GB | 4–6 min | ★★★★★ | ★★★★★ | ★★★★☆ | **Mejor calidad con GPU** |
| `mistral:7b` | 7B | 4.1 GB | 4–6 min | ★★★★☆ | ★★★★☆ | ★★★★☆ | GPU, buen equilibrio |
| `ministral:8b` | 8B | 4.9 GB | 5–7 min | ★★★★☆ | ★★★★☆ | ★★★★☆ | GPU, similar a mistral:7b |
| `gemma2:2b` | 2B | 1.6 GB | 50–80 s | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | CPU, alternativa a llama3.2:1b |
| `gemma2:9b` | 9B | 5.5 GB | 6–9 min | ★★★★☆ | ★★★★☆ | ★★★★☆ | GPU |
| `phi3.5` | 3.8B | 2.2 GB | 2–3 min | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | CPU, mejor JSON que llama3.2 |

#### Análisis por escenario

**Solo CPU (sin GPU):**
El cuello de botella es la velocidad. La negociación requiere varios turnos; con más de 3 minutos por turno el agente pierde el ritmo frente a oponentes más rápidos.
- Opción más rápida: `llama3.2:1b` o `gemma2:2b` (~1 min), asumiendo peor JSON
- Mejor equilibrio: `qwen2.5:3b` — sigue instrucciones JSON mejor que Llama 3.2 a velocidad similar
- A evitar: cualquier modelo ≥7B en CPU pura

**Con GPU (≥8 GB VRAM):**
La velocidad deja de ser problema. Priorizar calidad de seguimiento de instrucciones.
- Recomendado: `qwen2.5:7b` — el mejor probado para JSON estricto y restricciones complejas en español
- Alternativa: `mistral:7b` o `ministral:8b` — buena calidad, ligeramente más creativos en el texto

**Por qué Qwen 2.5 destaca para este caso:**
Qwen 2.5 fue entrenado específicamente con énfasis en output estructurado y seguimiento de system prompts. En benchmarks de JSON mode supera consistentemente a Llama 3.2 del mismo tamaño. Para un agente cuya corrección depende enteramente de generar JSON válido con restricciones, esto es determinante.

#### Cómo cambiar de modelo

1. Descargar el modelo con Ollama:
```powershell
ollama pull qwen2.5:3b
```

2. Editar `settings.toml`:
```toml
[LLM]
MODEL_NAME = "qwen2.5:3b"
BASE_URL = "http://127.0.0.1:11434/v1"
```

3. Reiniciar el agente:
```powershell
uv run fdi-pln-2614-p1
```
