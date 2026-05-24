import asyncio
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import ValidationError
import httpx

from fdi_pln_26XX_p1.core.config import settings
from fdi_pln_26XX_p1.agent.core_agent import agent, AgentDeps, OfferResponse

http_client: httpx.AsyncClient | None = None
historial_chats = {} 

async def revisar_correo_loop():
    butler_url = settings.BUTLER.ADDRESS
    mi_alias = settings.AGENT.ALIAS
    
    print("🔄 Iniciando motor autónomo...")
    ha_iniciado_contacto = False 
    
    while True:
        try:
            res_info = await http_client.get(f"{butler_url}/info?agente={mi_alias}")
            res_info.raise_for_status()
            estado = res_info.json()
            buzon = estado.get("Buzon", {})
            recursos = estado.get("Recursos", {})
            
            # 1. Iniciativa
            if not buzon and not ha_iniciado_contacto:
                res_gente = await http_client.get(f"{butler_url}/gente?agente={mi_alias}")
                if res_gente.status_code == 200:
                    gente = res_gente.json()
                    nombres = [g.get("alias") if isinstance(g, dict) else g for g in gente]
                    nombres = [n for n in nombres if n != mi_alias and n not in ["System", "Sistema"]]
                    
                    if nombres:
                        objetivo = nombres[0]
                        print(f"\n🎯 Tomando la iniciativa con {objetivo}...")
                        
                        deps = AgentDeps(inventory=recursos, budget=100.0, current_step=1, opponent_reputation=0.5)
                        if objetivo not in historial_chats:
                            historial_chats[objetivo] = []
                            
                        prompt_iniciativa = f"Inicia conversación con '{objetivo}'. Tienes: {recursos}. Propón un trato exacto."
                        
                        try:
                            result = await agent.run(prompt_iniciativa, deps=deps, message_history=historial_chats[objetivo])
                            historial_chats[objetivo] = result.all_messages()
                            
                            texto_crudo = result.output.strip()
                            inicio, fin = texto_crudo.find('{'), texto_crudo.rfind('}')
                            texto_json = texto_crudo[inicio:fin+1] if inicio != -1 and fin != -1 else texto_crudo
                            texto_json = re.sub(r',\s*}', '}', texto_json)
                            texto_json = re.sub(r'"\s*\+\s*"', '', texto_json)
                            texto_json = texto_json.replace('\\\n', '').replace('\\n\\', '\\n')
                                
                            datos_dict = json.loads(texto_json, strict=False)
                            respuesta_ia = OfferResponse.model_validate(datos_dict)
                            texto_iniciativa = respuesta_ia.msg # Acortado
                            
                        except Exception as e:
                            print(f"⚠️ Error iniciativa: {e}")
                            texto_iniciativa = f"Hola {objetivo}, tengo: {recursos}. ¿Qué ofreces exactamente?"

                        print(f"✉️ [ENVIADO]: {texto_iniciativa}")
                        await http_client.post(f"{butler_url}/carta?agente={mi_alias}", json={"remi": mi_alias, "dest": objetivo, "asunto": "Trueque", "cuerpo": texto_iniciativa})
                        ha_iniciado_contacto = True

            # --- NUEVO: FILTRO ANTI-SPAM (DRENAJE DE COLA) ---
            mensajes_a_procesar = {}
            for uid, mensaje in buzon.items():
                remitente, cuerpo = mensaje.get("remi"), mensaje.get("cuerpo")
                
                if remitente in [mi_alias, "System", "Sistema"]:
                    await http_client.delete(f"{butler_url}/mail/{uid}?agente={mi_alias}")
                    continue
                    
                # Si el profe mandó varios mensajes, borramos los viejos y nos quedamos solo con el último
                if remitente in mensajes_a_procesar:
                    uid_viejo = mensajes_a_procesar[remitente]["uid"]
                    await http_client.delete(f"{butler_url}/mail/{uid_viejo}?agente={mi_alias}")
                    print(f"🧹 Ignorando mensaje antiguo de {remitente} para evitar atascos.")
                    
                mensajes_a_procesar[remitente] = {"uid": uid, "cuerpo": cuerpo}
            # ------------------------------------------------

            # 2. Procesar Buzón Filtrado
            for remitente, datos in mensajes_a_procesar.items():
                uid = datos["uid"]
                cuerpo = datos["cuerpo"]
                
                print(f"\n📩 [NUEVO MENSAJE] de {remitente}: {cuerpo}")
                
                deps = AgentDeps(inventory=recursos, budget=100.0, current_step=1, opponent_reputation=0.5)
                if remitente not in historial_chats:
                    historial_chats[remitente] = []

                print("⚡ Llama 3 pensando...")
                result = await agent.run(cuerpo, deps=deps, message_history=historial_chats[remitente])
                
                historial_chats[remitente] = result.all_messages()
                # MEMORIA ULTRACORTA: Solo recordamos los últimos 2 mensajes
                if len(historial_chats[remitente]) > 2:
                    historial_chats[remitente] = historial_chats[remitente][-2:]
                
                texto_crudo = result.output.strip()
                inicio, fin = texto_crudo.find('{'), texto_crudo.rfind('}')
                texto_json = texto_crudo[inicio:fin+1] if inicio != -1 and fin != -1 else texto_crudo
                texto_json = re.sub(r',\s*}', '}', texto_json)
                texto_json = re.sub(r'"\s*\+\s*"', '', texto_json)
                texto_json = texto_json.replace('\\\n', '').replace('\\n\\', '\\n')
                
                try:
                    datos_dict = json.loads(texto_json, strict=False)
                    respuesta_ia = OfferResponse.model_validate(datos_dict)
                    texto_respuesta = respuesta_ia.msg # Acortado
                    
                    if getattr(respuesta_ia, 'env', None): # Acortado y verificando si hay paquete
                        print(f"📦 ¡TRATO CERRADO! Enviando paquete a {remitente}: {respuesta_ia.env}")
                        await http_client.post(f"{butler_url}/paquete/{remitente}?agente={mi_alias}", json=respuesta_ia.env)
                        historial_chats[remitente] = []
                            
                except Exception as e:
                    print(f"⚠️ Error de parseo ({e}).")
                    # En caso de error, mandamos un mensaje cerrado que no invite al profe a seguir hablando
                    texto_respuesta = "Lo siento, mi almacén está cerrado por inventario. Hablaremos luego."
                    historial_chats[remitente] = [] # Borramos memoria para destrabar
                    
                print(f"✉️ [RESPUESTA] a {remitente}: {texto_respuesta}")
                await http_client.post(f"{butler_url}/carta?agente={mi_alias}", json={"remi": mi_alias, "dest": remitente, "asunto": "Respuesta", "cuerpo": texto_respuesta})
                await http_client.delete(f"{butler_url}/mail/{uid}?agente={mi_alias}")
                print("🗑️ Mensaje borrado.")
                
        except Exception as e:
            print(f"⚠️ Error procesando turno: {e}")
            await asyncio.sleep(10)
            continue
            
        await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    butler_url = settings.BUTLER.ADDRESS
    mi_alias = settings.AGENT.ALIAS
    
    try:
        await http_client.post(f"{butler_url}/alias/{mi_alias}?agente={mi_alias}")
        print(f"¡Puesto '{mi_alias}' registrado con éxito!")
        tarea_correo = asyncio.create_task(revisar_correo_loop())
    except Exception as e:
        print(f"ADVERTENCIA: Fallo al conectar. Error: {e}")

    yield
    
    if 'tarea_correo' in locals():
        tarea_correo.cancel()
    await http_client.aclose()

app = FastAPI(title="Agente de Trueque", lifespan=lifespan)