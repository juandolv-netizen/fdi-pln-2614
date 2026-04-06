```markdown
# Práctica 4: Aplicación de Búsqueda y RAG (PLN)

## Integrantes del Equipo (Grupo 2614)
* [Nombre y Apellidos del integrante 1]
* [Nombre y Apellidos del integrante 2]
* [Añadir el resto de integrantes]

## Descripción
Aplicación de terminal interactiva (TUI) que procesa un archivo fuente HTML, lo divide en fragmentos (chunks) y permite realizar consultas sobre el texto utilizando técnicas de procesamiento de lenguaje natural y modelos de lenguaje.

## Requisitos del Sistema
* **Python**: >= 3.13
* **Gestor de paquetes**: `uv`
* **Dependencias de IA**: `ollama` instalado y en ejecución local con el modelo `llama3.2` descargado (exclusivo para el modo RAG).

## Instalación
Sincronizar el entorno virtual y descargar las dependencias exactas definidas en `pyproject.toml` y `uv.lock`:

```bash
uv sync
```

Descargar el modelo LLM local (necesario para la ejecución del RAG):
```bash
ollama pull llama3.2
```

## Ejecución

**Modo completamente interactivo (Menú Principal):**
```bash
uv run fdi-pln-2614-p4
```

**Modo con paso de argumentos directos:**
```bash
uv run fdi-pln-2614-p4 [ruta_al_archivo.html] -q "tu consulta aquí" -m [clasica|semantica|rag]
```
*Si se omite la ruta del archivo, el sistema asume por defecto `2000-h.htm` en el directorio de ejecución.*

## Modos de Operación
1. **Búsqueda clásica (`clasica`)**: Recuperación léxica. Aplica lematización, eliminación de signos de puntuación y *stopwords* mediante spaCy (`es_core_news_md`). Ordena los resultados empleando un sistema de ranking TF-IDF calculado desde cero.
2. **Búsqueda semántica (`semantica`)**: Recuperación densa. Genera *embeddings* vectoriales para la consulta y los fragmentos del texto mediante spaCy. Calcula la relevancia empleando similitud del coseno con `numpy`.
3. **RAG (`rag`)**: Sistema *Retrieval-Augmented Generation*. Realiza una **búsqueda híbrida** combinando y normalizando las puntuaciones léxicas (TF-IDF) y semánticas (Coseno). Los mejores resultados se inyectan como contexto estructurado en el modelo `llama3.2` a través de Ollama para sintetizar una respuesta natural que incluye citas y referencias directas a los fragmentos recuperados.

## Preprocesado del Dataset
El archivo original proporcionado (`2000-h.htm`, proveniente de Project Gutenberg) presentaba una estructura HTML orientada a la maquetación visual, lo que dificultaba la extracción jerárquica de la información. Se generó un archivo de trabajo (`2000-h.htm`) aplicando el siguiente preprocesado manual:

1. **Eliminación de ruido (Boilerplate):** Se eliminaron las cabeceras legales, metadatos, licencias de Project Gutenberg y pies de página para evitar que este texto irrelevante contaminara los fragmentos indexados y los resultados de búsqueda.
2. **Estandarización de jerarquía (Etiquetas HTML):** Se reestructuraron las etiquetas de encabezado para reflejar la anatomía real de la obra y alimentar correctamente la máquina de estados del *scraper* (BeautifulSoup):
   * Se asignó la etiqueta `<h2>` exclusivamente a las grandes divisiones del libro (ej. "Primera Parte", "Segunda Parte").
   * Se asignó la etiqueta `<h3>` a los títulos de los capítulos y textos preliminares (ej. "Capítulo I", "Prólogo").
3. **Limpieza estructural:** Corrección de anidamientos incorrectos y etiquetas de párrafo (`<p>`) rotas para asegurar una lectura lineal ininterrumpida y una correcta generación de los *chunks*.