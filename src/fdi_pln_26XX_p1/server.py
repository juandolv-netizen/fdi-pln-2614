import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import ValidationError
import httpx

from fdi_pln_26XX_p1.core.config import settings
from fdi_pln_26XX_p1.agent.core_agent import agent, AgentDeps, OfferResponse

http_client: httpx.AsyncClient | None = None

historial_chats = {}


async def revisar_correo_loop():
    """Bucle infinito que revisa el buzón y toma la iniciativa."""
    butler_url = settings.BUTLER.ADDRESS
    mi_alias = settings.AGENT.ALIAS

    print("🔄 Iniciando motor autónomo de revisión de correo...")
    ha_iniciado_contacto = False

    while True:
        try:
            res_info = await http_client.get(f"{butler_url}/info?agente={mi_alias}")
            res_info.raise_for_status()
            estado = res_info.json()
            buzon = estado.get("Buzon", {})
            recursos = estado.get("Recursos", {})

            # 1. Lógica Proactiva
            if not buzon and not ha_iniciado_contacto:
                res_gente = await http_client.get(
                    f"{butler_url}/gente?agente={mi_alias}"
                )
                if res_gente.status_code == 200:
                    gente = res_gente.json()
                    nombres = [
                        g.get("alias") if isinstance(g, dict) else g for g in gente
                    ]
                    nombres = [
                        n
                        for n in nombres
                        if n != mi_alias and n not in ["System", "Sistema"]
                    ]

                    if nombres:
                        objetivo = nombres[0]
                        print(
                            f"\n🎯 ¡Tomando la iniciativa! Dejando que Llama 3 piense cómo saludar a {objetivo}..."
                        )

                        deps = AgentDeps(
                            inventory=recursos,
                            budget=100.0,
                            current_step=1,
                            opponent_reputation=0.5,
                        )
                        if objetivo not in historial_chats:
                            historial_chats[objetivo] = []

                        prompt_iniciativa = f"Inicia una conversación con el agente '{objetivo}' para proponer un trueque. Tienes estos recursos disponibles: {recursos}. ¿Qué le dices para romper el hielo y buscar un acuerdo?"

                        try:
                            result = await agent.run(
                                prompt_iniciativa,
                                deps=deps,
                                message_history=historial_chats[objetivo],
                            )
                            historial_chats[objetivo] = result.all_messages()
                            texto_crudo = result.output.strip()
                            inicio = texto_crudo.find("{")
                            fin = texto_crudo.rfind("}")

                            if inicio != -1 and fin != -1:
                                texto_json = texto_crudo[inicio : fin + 1]
                            else:
                                texto_json = texto_crudo

                            respuesta_ia = OfferResponse.model_validate_json(texto_json)
                            print(
                                f"💭 Análisis interno IA (Iniciativa): {respuesta_ia.analisis_interno}"
                            )
                            texto_iniciativa = respuesta_ia.mensaje_respuesta

                        except (ValidationError, ValueError):
                            print(
                                "⚠️ Llama 3 falló al estructurar la iniciativa. Usando mensaje por defecto."
                            )
                            texto_iniciativa = f"Hola {objetivo}, mi inventario es: {recursos}. ¿Tienes alguna oferta para mí?"

                        print(f"✉️ [MENSAJE ENVIADO]: {texto_iniciativa}")

                        carta_inicial = {
                            "remi": mi_alias,
                            "dest": objetivo,
                            "asunto": "Iniciando trueque",
                            "cuerpo": texto_iniciativa,
                        }
                        await http_client.post(
                            f"{butler_url}/carta?agente={mi_alias}", json=carta_inicial
                        )
                        ha_iniciado_contacto = True

            # 2. Procesar el buzón con MEMORIA
            for uid, mensaje in buzon.items():
                remitente = mensaje.get("remi")
                cuerpo = mensaje.get("cuerpo")

                if remitente in [mi_alias, "System", "Sistema"]:
                    print(f"🧹 Notificación automática de {remitente} leída y borrada.")
                    await http_client.delete(
                        f"{butler_url}/mail/{uid}?agente={mi_alias}"
                    )
                    continue

                print(f"\n📩 [NUEVO MENSAJE] de {remitente}: {cuerpo}")

                deps = AgentDeps(
                    inventory=recursos,
                    budget=100.0,
                    current_step=1,
                    opponent_reputation=0.5,
                )

                if remitente not in historial_chats:
                    historial_chats[remitente] = []

                print("🧠 Llama 3 está pensando y recordando la conversación...")

                result = await agent.run(
                    cuerpo, deps=deps, message_history=historial_chats[remitente]
                )

                historial_chats[remitente] = result.all_messages()

                texto_crudo = result.output.strip()
                inicio = texto_crudo.find("{")
                fin = texto_crudo.rfind("}")

                if inicio != -1 and fin != -1:
                    texto_json = texto_crudo[inicio : fin + 1]
                else:
                    texto_json = texto_crudo

                try:
                    # 1. Usamos json.loads con strict=False para perdonar los "Enters"
                    datos_dict = json.loads(texto_json, strict=False)
                    # 2. Pasamos el diccionario limpio a Pydantic
                    respuesta_ia = OfferResponse.model_validate(datos_dict)

                    print(f"💭 Análisis interno IA: {respuesta_ia.analisis_interno}")
                    texto_respuesta = respuesta_ia.mensaje_respuesta

                except Exception as e:
                    print(
                        f"⚠️ Error de parseo ({e}). Texto extraído: {texto_json[:150]}..."
                    )
                    texto_respuesta = "Disculpa, mi cerebro estaba procesando el inventario y no te entendí bien. ¿Podemos retomar la negociación? ¿Qué me ofreces exactamente?"

                print(f"✉️ [ENVIANDO RESPUESTA] a {remitente}: {texto_respuesta}")

                nueva_carta = {
                    "remi": mi_alias,
                    "dest": remitente,
                    "asunto": "Respuesta sobre el trueque",
                    "cuerpo": texto_respuesta,
                }
                await http_client.post(
                    f"{butler_url}/carta?agente={mi_alias}", json=nueva_carta
                )
                await http_client.delete(f"{butler_url}/mail/{uid}?agente={mi_alias}")
                print("🗑️ Mensaje borrado del buzón.")

        except Exception as e:
            print(f"⚠️ Error procesando turno: {e}")
            await asyncio.sleep(20)
            continue

        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)

    butler_url = settings.BUTLER.ADDRESS
    mi_alias = settings.AGENT.ALIAS

    try:
        # Registrar puesto
        await http_client.post(f"{butler_url}/alias/{mi_alias}?agente={mi_alias}")
        print(f"¡Puesto '{mi_alias}' registrado con éxito!")

        # Arrancar la tarea en segundo plano
        tarea_correo = asyncio.create_task(revisar_correo_loop())

    except Exception as e:
        print(f"ADVERTENCIA: Fallo al comunicarse inicial con el Butler. Error: {e}")

    yield

    # Limpieza al apagar
    if "tarea_correo" in locals():
        tarea_correo.cancel()
    await http_client.aclose()
    print("Conexiones cerradas limpiamente.")


app = FastAPI(title="Agente de Trueque", lifespan=lifespan)
