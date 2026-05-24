# Práctica 1: Agente de Negociación PLN
**Equipo XX** (Sustituye XX por tu número)
**Integrantes:** Juan (Añade tus apellidos y los de tus compañeros si los hay)

## Descripción
Agente conversacional desarrollado para la Práctica 1. Utiliza un enfoque híbrido:
1. **Lógica Asíncrona:** Bucle continuo con `asyncio` y `httpx` para revisar el buzón sin bloquear el servidor.
2. **Motor LLM Local:** Utiliza `Llama 3.2` conectado a través de `pydantic-ai`.
3. **Parseo Robusto:** El agente utiliza un prompt estricto y extracción matemática de llaves `{}` combinada con `json.loads(..., strict=False)` para garantizar que las alucinaciones de formato de Llama 3 no rompan la validación de Pydantic.
4. **Proactividad y Memoria:** El agente escanea el servidor para dar el primer paso si su buzón está vacío e implementa un diccionario de historial para mantener el contexto de la conversación con cada contrincante.

## Dependencias
Se ha utilizado el stack permitido en la guía. *Nota técnica: Se utiliza la clase OpenAIProvider/AsyncOpenAI internamente para enrutar las peticiones de `pydantic-ai` hacia el host local de Ollama en el puerto 11434.*

## Ejecución
Para arrancar el agente en el entorno Linux de los laboratorios:
```bash
uv run fdi-pln-26XX-p1