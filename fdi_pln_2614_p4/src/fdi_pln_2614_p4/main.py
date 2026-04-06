import re
import os
import math
import argparse
import sys
import spacy
import numpy as np
from numpy.linalg import norm
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
import ollama

# Configuración de inicialización
console = Console()

with console.status("[bold cyan]Cargando modelo spaCy (md)...[/bold cyan]", spinner="dots"):
    nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])

# --- CONSTANTES ---
TAMANO_CHUNK = 150
SOLAPAMIENTO = 50


def lematizar_texto(texto: str) -> str:
    """
    Procesa el texto a minúsculas, extrae los lemas y elimina puntuación y stopwords.
    """
    doc = nlp(texto.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])


def cargar_documento(ruta_archivo: str) -> BeautifulSoup:
    """
    Lee un archivo local HTML y retorna su representación parseada con BeautifulSoup.
    Termina la ejecución si no encuentra el archivo.
    """
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as file:
            return BeautifulSoup(file.read(), 'html.parser')
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] No se encontró el archivo '{ruta_archivo}'.")
        sys.exit(1)


def extraer_textos(soup: BeautifulSoup) -> tuple[list[str], list[dict]]:
    """
    Extrae el texto secuencialmente guardando estado de la Parte (h2) y Título (h3) correspondientes.
    """
    elementos = soup.find_all(['p', 'h3', 'h2', 'li'])
    palabras_totales = []
    mapping_metadatos = []

    parte_actual = "Sin parte"
    titulo_actual = "Sin título previo"

    for el in elementos:
        if el.name == 'h2':
            parte_actual = el.get_text(strip=True)
            continue
        if el.name == 'h3':
            titulo_actual = el.get_text(strip=True)
            continue
        
        texto = el.get_text(" ", strip=True)
        if not texto: 
            continue
        
        palabras = texto.split()
        palabras_totales.extend(palabras)
        mapping_metadatos.extend([{'titulo': titulo_actual, 'parte': parte_actual}] * len(palabras))

    return palabras_totales, mapping_metadatos


def generar_chunks(palabras_totales: list[str], mapping_metadatos: list[dict]) -> list[dict]:
    """
    Agrupa palabras e inyecta la Parte y el Título en cada chunk.
    """
    chunks = []
    for i in range(0, len(palabras_totales), TAMANO_CHUNK - SOLAPAMIENTO):
        ventana = palabras_totales[i : i + TAMANO_CHUNK]
        if not ventana: 
            break
        
        texto_original = " ".join(ventana)
        chunks.append({
            'titulo': mapping_metadatos[i]['titulo'],
            'parte': mapping_metadatos[i]['parte'],
            'texto_original': texto_original,
            'texto_lema': lematizar_texto(texto_original) 
        })
    return chunks


def buscar_tfidf(chunks: list[dict], query_lema: str) -> list[dict]:
    """
    Ejecuta una búsqueda clásica con ranking TF-IDF sobre los chunks dados.
    """
    coincidencias = []
    textos_vistos = set() 
    query_tokens = set(query_lema.split())
    
    # Calcular IDF
    N = len(chunks)
    df = {token: 0 for token in query_tokens}
    
    for c in chunks:
        chunk_tokens_set = set(c['texto_lema'].split())
        for token in query_tokens:
            if token in chunk_tokens_set:
                df[token] += 1
                
    idf = {token: math.log(N / (df[token] if df[token] > 0 else 1)) for token in query_tokens}

    # Calcular TF y Score
    for c in chunks:
        chunk_tokens = c['texto_lema'].split()
        total_terms = len(chunk_tokens)
        score_tfidf = 0
        
        if total_terms > 0:
            for token in query_tokens:
                tf = chunk_tokens.count(token) / total_terms
                score_tfidf += tf * idf[token]
        
        if score_tfidf > 0 and c['texto_original'] not in textos_vistos:
            c['score'] = round(score_tfidf, 4)
            coincidencias.append(c)
            textos_vistos.add(c['texto_original'])

    coincidencias.sort(key=lambda x: x['score'], reverse=True)
    return coincidencias


def calcular_similitud_coseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula la similitud del coseno entre dos vectores."""
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))


def buscar_semantica(chunks: list[dict], query: str) -> list[dict]:
    """
    Ejecuta una búsqueda semántica usando embeddings de spaCy y similitud del coseno.
    """
    query_vec = nlp(query).vector
    coincidencias = []
    textos_vistos = set()

    for c in chunks:
        if 'vector' not in c:
            c['vector'] = nlp(c['texto_original']).vector
        
        similitud = calcular_similitud_coseno(query_vec, c['vector'])
        
        if similitud > 0.4 and c['texto_original'] not in textos_vistos:
            c['score'] = round(similitud, 4)
            coincidencias.append(c)
            textos_vistos.add(c['texto_original'])

    coincidencias.sort(key=lambda x: x['score'], reverse=True)
    return coincidencias


def buscar_hibrida(chunks: list[dict], query_lema: str, query_original: str) -> list[dict]:
    """
    Combina las puntuaciones de TF-IDF y Similitud del Coseno normalizándolas a una escala común.
    """
    clasicas = buscar_tfidf(chunks, query_lema)
    semanticas = buscar_semantica(chunks, query_original)

    max_tfidf = max((c['score'] for c in clasicas), default=1.0)
    max_sem = max((c['score'] for c in semanticas), default=1.0)

    combinados = {}
    
    for c in clasicas:
        c_copy = c.copy()
        c_copy['score_hibrido'] = (c['score'] / max_tfidf) * 0.5
        combinados[c['texto_original']] = c_copy

    for s in semanticas:
        txt = s['texto_original']
        if txt in combinados:
            combinados[txt]['score_hibrido'] += (s['score'] / max_sem) * 0.5
        else:
            s_copy = s.copy()
            s_copy['score_hibrido'] = (s['score'] / max_sem) * 0.5
            combinados[txt] = s_copy

    resultado = list(combinados.values())
    for r in resultado:
        r['score'] = round(r['score_hibrido'], 4)

    resultado.sort(key=lambda x: x['score'], reverse=True)
    return resultado


def agrupar_capitulos(coincidencias: list[dict]) -> tuple[dict, list[str], int]:
    """
    Estandariza los capítulos, distingue por Parte y separa los textos preliminares al final.
    Retorna el diccionario agrupado, la lista de títulos y el índice de corte para la UI.
    """
    agrupados_caps = {}
    agrupados_prelims = {}

    for item in coincidencias:
        titulo_orig = item['titulo']
        parte_orig = item.get('parte', '')

        parte_limpia = ""
        if re.search(r'(?i)primera|1', parte_orig):
            parte_limpia = "Primera Parte"
        elif re.search(r'(?i)segunda|2', parte_orig):
            parte_limpia = "Segunda Parte"
        else:
            parte_limpia = parte_orig

        titulo_orig = re.sub(r'(?i)cap[íi]tulo\s+primero', 'Capítulo I', titulo_orig)
        match = re.search(r'(?i)cap[íi]tulo\s+([IVXLCDM]+)', titulo_orig)

        if match:
            titulo_limpio = f"{parte_limpia} - Capítulo {match.group(1).upper()}"
            item['titulo_limpio'] = titulo_limpio

            if titulo_limpio not in agrupados_caps:
                agrupados_caps[titulo_limpio] = []
            agrupados_caps[titulo_limpio].append(item)
        else:
            titulo_limpio = f"{parte_limpia} - {titulo_orig}"
            item['titulo_limpio'] = titulo_limpio

            if titulo_limpio not in agrupados_prelims:
                agrupados_prelims[titulo_limpio] = []
            agrupados_prelims[titulo_limpio].append(item)

    titulos_caps_ord = sorted(agrupados_caps.keys(), key=lambda t: max(r['score'] for r in agrupados_caps[t]), reverse=True)
    titulos_prelims_ord = sorted(agrupados_prelims.keys(), key=lambda t: max(r['score'] for r in agrupados_prelims[t]), reverse=True)

    agrupados = {**agrupados_caps, **agrupados_prelims}
    titulos_ordenados = titulos_caps_ord + titulos_prelims_ord
    
    return agrupados, titulos_ordenados, len(titulos_caps_ord)


def generar_respuesta_rag(query: str, resultados: list[dict]) -> str:
    """
    Utiliza Ollama para generar una respuesta basada en el contexto, 
    permitiendo síntesis y parafraseo controlado para evitar respuestas vacías.
    """
    if not resultados:
        return "No se encontró contexto relevante para responder."

    # Mantenemos un contexto amplio (10 fragmentos)
    contexto = "\n\n".join([f"[{i+1}] ({r['titulo_limpio']}): {r['texto_original']}" for i, r in enumerate(resultados[:10])])
    
    prompt = f"""Responde a la consulta del usuario basándote ÚNICAMENTE en el siguiente contexto.

Instrucciones:
1. Sintetiza y redacta una respuesta coherente relacionando los fragmentos recuperados con la consulta.
2. Tienes permitido interpretar el contexto para construir la respuesta (por ejemplo, si el texto habla de 'los de a caballo' en 'la noche', puedes conectarlo con la consulta), pero TIENES ESTRICTAMENTE PROHIBIDO inventar eventos, tramas o personajes que no estén en el texto.
3. Si los fragmentos mencionan eventos, lugares o descripciones que se acercan al tema de la consulta, descríbelos detalladamente basándote en lo que lees.
4. Al final de cada idea o escena descrita, incluye la referencia al fragmento usado (ejemplo: [1], [3]).
5. Si no hay una respuesta directa, explica de forma útil qué es lo que sí ocurre en los fragmentos en relación a los términos buscados.

Contexto:
{contexto}

Consulta: {query}"""

    try:
        response = ollama.chat(
            model='llama3.2', 
            messages=[
                {
                    'role': 'system', 
                    'content': 'Eres un asistente experto en analizar textos literarios. Tu tarea es construir una respuesta útil, descriptiva y bien redactada utilizando solo el contexto proporcionado, sin alucinar información externa.'
                },
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.3, # Ligera subida para permitir fluidez y conexión de conceptos
                'top_p': 0.5
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"[red]Error al conectar con Ollama:[/] {e}\n(Asegúrate de ejecutar 'ollama run llama3.2' en otra terminal)."


def resaltar_texto(texto: str, query: str) -> str:
    """
    Colorea de amarillo las raíces de las palabras buscadas usando expresiones regulares.
    """
    tokens = [t for t in query.split() if len(t) > 3] 
    texto_resaltado = texto
    for t in tokens:
        texto_resaltado = re.sub(rf'(?i)(\b{re.escape(t)}\w*\b)', r'[bold yellow]\1[/bold yellow]', texto_resaltado)
    return texto_resaltado


def interfaz_interactiva(agrupados: dict, titulos_ordenados: list[str], corte_preliminares: int, query_original: str):
    """
    Despliega la TUI. Añadida navegación multinivel para retornar al menú principal con 'q'.
    """
    LIMIT_CAPITULOS = 15
    titulos_mostrar = titulos_ordenados[:LIMIT_CAPITULOS]
    
    while True:
        console.clear()
        
        tabla_capitulos = Table(title=f"Top {len(titulos_mostrar)} secciones más relevantes (de {len(titulos_ordenados)} en total)", header_style="bold magenta")
        tabla_capitulos.add_column("Opción", style="cyan", justify="center")
        tabla_capitulos.add_column("Sección", style="white")
        tabla_capitulos.add_column("Resultados", justify="right", style="yellow")
        tabla_capitulos.add_column("Score Máximo", justify="right", style="green")

        for i, titulo in enumerate(titulos_mostrar, start=1):
            if corte_preliminares > 0 and i == corte_preliminares + 1:
                tabla_capitulos.add_section()
                
            max_score = max(r['score'] for r in agrupados[titulo])
            estilo_titulo = "white" if i <= corte_preliminares else "italic bright_black"
            
            tabla_capitulos.add_row(str(i), f"[{estilo_titulo}]{titulo}[/]", str(len(agrupados[titulo])), str(max_score))
        
        console.print(tabla_capitulos)
        
        opcion_str = Prompt.ask("\n[bold]Selecciona la sección (o [red]'q'[/red] para el menú principal)[/bold]", default="q", show_default=False)
        if opcion_str.lower() in ['q', 'quit', 'salir']:
            return  # Retorna al bucle principal en main()
            
        try:
            opcion = int(opcion_str)
            if 1 <= opcion <= len(titulos_mostrar):
                titulo_sel = titulos_mostrar[opcion - 1]
                resultados = agrupados[titulo_sel]

                if len(resultados) == 1:
                    sel = resultados[0]
                    console.clear()
                    texto_resaltado = resaltar_texto(sel['texto_original'], query_original)
                    panel = Panel(texto_resaltado, title=f"[bold cyan]{sel['titulo_limpio']}[/] - Score: [green]{sel['score']}[/]", border_style="cyan")
                    console.print(panel)
                    Prompt.ask("\n[bold]Presiona Enter para volver[/bold]")
                    continue

                while True:
                    console.clear()
                    tabla_pasajes = Table(title=f"Resultados en '{titulo_sel}'", header_style="bold magenta")
                    tabla_pasajes.add_column("Opción", style="cyan", justify="center")
                    tabla_pasajes.add_column("Score", justify="right", style="green")
                    tabla_pasajes.add_column("Resumen del pasaje", style="white")

                    for j, res in enumerate(resultados, start=1):
                        palabras = res['texto_original'].split()
                        resumen = " ".join(palabras[:12]) + ("..." if len(palabras) > 12 else "")
                        tabla_pasajes.add_row(str(j), str(res['score']), resaltar_texto(resumen, query_original))

                    console.print(tabla_pasajes)
                    
                    sub_op = Prompt.ask("\n[bold]Selecciona el pasaje ([yellow]'v'[/yellow] volver, [red]'q'[/red] menú principal)[/bold]", default="v", show_default=False)
                    if sub_op.lower() in ['v', 'volver']:
                        break
                    elif sub_op.lower() in ['q', 'quit', 'salir']:
                        return  # Salida total hacia el menú principal
                    
                    sub_opcion = int(sub_op)
                    if 1 <= sub_opcion <= len(resultados):
                        sel = resultados[sub_opcion - 1]
                        console.clear()
                        texto_resaltado = resaltar_texto(sel['texto_original'], query_original)
                        panel = Panel(texto_resaltado, title=f"[bold cyan]{sel['titulo_limpio']}[/] - Score: [green]{sel['score']}[/]", border_style="cyan")
                        console.print(panel)
                        Prompt.ask("\n[bold]Presiona Enter para volver[/bold]")
                    else:
                        console.print("[bold red]Pasaje fuera de rango.[/bold red]")
                        Prompt.ask("\n[bold]Presiona Enter para continuar[/bold]")
            else:
                console.print("[bold red]Sección fuera de rango.[/bold red]")
                Prompt.ask("\n[bold]Presiona Enter para continuar[/bold]")
        except ValueError:
            console.print("[bold red]Entrada inválida.[/bold red]")
            Prompt.ask("\n[bold]Presiona Enter para continuar[/bold]")


def main():
    # 1. Configurar argumentos de consola
    parser = argparse.ArgumentParser(description="Buscador inteligente PLN")
    ruta_defecto = os.path.join(os.path.dirname(__file__), "2000-h.htm")
    parser.add_argument("archivo", nargs="?", default=ruta_defecto, help="Ruta al archivo HTML")
    parser.add_argument("-q", "--query", help="Texto a buscar.", default=None)
    parser.add_argument("-m", "--modo", choices=['clasica', 'semantica', 'rag'], default=None, help="Modo de operación")
    args = parser.parse_args()

    # 2. Carga inicial pesada (ejecutada solo una vez)
    with console.status("[bold cyan]Cargando documento y generando chunks...[/bold cyan]", spinner="dots"):
        soup = cargar_documento(args.archivo)
        palabras_totales, mapping_metadatos = extraer_textos(soup)
        chunks = generar_chunks(palabras_totales, mapping_metadatos)

    modo_actual = args.modo
    query_actual = args.query

    # 3. Bucle principal de la aplicación
    while True:
        console.clear()

        # Menú interactivo
        if not modo_actual:
            console.print("\n[bold magenta]--- MENÚ DE OPERACIÓN ---[/bold magenta]")
            console.print("1. Búsqueda clásica")
            console.print("2. Búsqueda semántica")
            console.print("3. RAG")
            console.print("0. [red]Salir de la aplicación[/red]")
            
            while True:
                opcion = console.input("\n[bold]Selecciona una opción (0-3): [/bold]")
                if opcion == '1':
                    modo_actual = 'clasica'
                    break
                elif opcion == '2':
                    modo_actual = 'semantica'
                    break
                elif opcion == '3':
                    modo_actual = 'rag'
                    break
                elif opcion == '0':
                    console.print("[bold green]Saliendo del buscador...[/bold green]")
                    sys.exit(0)
                console.print("[bold red]Opción inválida.[/bold red]")

        # Entrada de consulta
        if not query_actual:
            query_actual = Prompt.ask("\n[bold]Introduce el texto a buscar (o [red]'q'[/red] para volver al menú)[/bold]")
            if query_actual.lower() in ['q', 'quit', 'salir']:
                modo_actual = None
                continue

        # Procesamiento
        with console.status("[bold cyan]Procesando consulta y ejecutando búsqueda...[/bold cyan]", spinner="dots"):
            query_lema = lematizar_texto(query_actual)

            if modo_actual == 'clasica':
                coincidencias = buscar_tfidf(chunks, query_lema)
            elif modo_actual == 'semantica':
                coincidencias = buscar_semantica(chunks, query_actual)
            elif modo_actual == 'rag':
                coincidencias = buscar_hibrida(chunks, query_lema, query_actual)
            else:
                console.print(f"[bold yellow]Advertencia:[/] El modo '{modo_actual}' aún no está implementado.")
                modo_actual = None
                query_actual = None
                Prompt.ask("\n[bold]Presiona Enter para continuar[/bold]")
                continue

        if not coincidencias:
            console.print("[bold yellow]No se encontraron resultados para tu búsqueda.[/bold yellow]")
            modo_actual = None
            query_actual = None
            Prompt.ask("\n[bold]Presiona Enter para volver al menú[/bold]")
            continue

        agrupados, titulos_ordenados, corte_preliminares = agrupar_capitulos(coincidencias)

        if modo_actual == 'rag':
            with console.status("[bold magenta]Generando respuesta con Ollama (RAG)...[/bold magenta]", spinner="bouncingBar"):
                respuesta = generar_respuesta_rag(query_actual, coincidencias)
            
            console.clear()
            console.print(Panel(respuesta, title="[bold magenta]Respuesta RAG[/bold magenta]", border_style="magenta"))
            Prompt.ask("\n[bold]Presiona Enter para explorar los resultados detallados en la tabla[/bold]")

        interfaz_interactiva(agrupados, titulos_ordenados, corte_preliminares, query_actual)
        
        # Purga de estado para iniciar un nuevo ciclo limpio
        modo_actual = None
        query_actual = None


if __name__ == "__main__":
    main()