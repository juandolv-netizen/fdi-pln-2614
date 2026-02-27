from dataclasses import dataclass
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from fdi_pln_26XX_p1.core.config import settings


@dataclass
class AgentDeps:
    inventory: dict[str, int]
    budget: float
    current_step: int
    opponent_reputation: float


class OfferResponse(BaseModel):
    analisis_interno: str = Field(
        description="Piensa qué necesitas, qué ofreces y tu estrategia."
    )
    mensaje_respuesta: str = Field(
        description="Respuesta natural y conversacional para el otro agente."
    )


ollama_client = AsyncOpenAI(
    base_url=settings.LLM.BASE_URL, api_key="ollama", timeout=180.0
)

model = OpenAIModel(
    model_name=settings.LLM.MODEL_NAME,
    provider=OpenAIProvider(openai_client=ollama_client),
)

# Agente
agent = Agent(
    model=model,
    deps_type=AgentDeps,
    system_prompt=(
        "Eres Npi, un comerciante astuto y amigable en un mercado de trueque.\n"
        "Tu objetivo es conseguir los recursos que te faltan intercambiando los que te sobran.\n"
        "Tu inventario actual se te pasará en los datos del sistema.\n"
        "REGLAS INQUEBRANTABLES:\n"
        "1. NO escribas texto fuera del formato JSON solicitado.\n"
        "2. En 'mensaje_respuesta', habla con fluidez y naturalidad.\n"
        "3. Sé directo pero educado.\n"
        "4. Solo intercambias recursos en cantidades ENTERAS (1, 2, 3...).\n"
        "5. PROHIBIDO USAR SALTOS DE LÍNEA REALES (Enter) DENTRO DEL JSON. Si quieres hacer párrafos en el mensaje, escribe los caracteres '\\n' literalmente.\n"
        "FORMATO EXACTO ESPERADO:\n"
        "{\n"
        '  "analisis_interno": "Me pide telas pero no tengo. Le ofreceré madera a cambio de su oro.",\n'
        '  "mensaje_respuesta": "Hola Prof,\\n\\nLamentablemente no tengo telas ahora mismo. Sin embargo, veo que tengo madera de sobra. ¿Te interesaría cambiar tu oro por mi madera?"\n'
        "}"
    ),
)
