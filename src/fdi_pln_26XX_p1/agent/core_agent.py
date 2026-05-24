from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field, field_validator
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

# --- MICRO-MODELO (Máxima velocidad de generación) ---
class OfferResponse(BaseModel):
    msg: str = Field(description="Respuesta super directa y breve.")
    env: dict[str, int] | None = Field(default=None, description="Ej: {'oro': 2}. Null si no hay trato.")

    @field_validator('env', mode='before')
    @classmethod
    def fix_empty_values(cls, v: Any):
        if not v or v == "null" or v == "None" or v == "":
            return None
        return v

ollama_client = AsyncOpenAI(
    base_url=settings.LLM.BASE_URL, api_key="ollama", timeout=180.0
)

model = OpenAIModel(
    model_name=settings.LLM.MODEL_NAME,
    provider=OpenAIProvider(openai_client=ollama_client),
)

agent = Agent(
    model=model,
    deps_type=AgentDeps,
    system_prompt=(
        "Eres Npi, un comerciante de trueque súper directo.\n"
        "Inventario: {inventory}\n"
        "REGLAS:\n"
        "1. SOLO output en JSON.\n"
        "2. En 'msg', SÉ EXTREMADAMENTE ESPECÍFICO. Usa SIEMPRE esta estructura exacta: 'Te doy [N] [recurso] a cambio de [N] [recurso]'.\n"
        "3. NO uses saltos de línea reales. Usa '\\n'.\n"
        "4. ¡PELIGRO DE ESTAFA! NUNCA rellenes 'env' al proponer un trato.\n"
        "5. SOLO rellena 'env' cuando el otro agente ACEPTA explícitamente tu oferta o dice 'Acuerdo'.\n"
        "FORMATO 1 - PROPONER:\n"
        "{\n"
        '  "msg": "Te doy 2 oro a cambio de 1 tela. ¿Aceptas?",\n'
        '  "env": null\n'
        "}\n"
        "FORMATO 2 - PAGAR:\n"
        "{\n"
        '  "msg": "Perfecto, trato cerrado.",\n'
        '  "env": {"oro": 2}\n'
        "}"
    )
)