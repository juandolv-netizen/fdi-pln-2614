import json
import re
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from fdi_pln_2614_p1.agent.classifier import MessagePhase
from fdi_pln_2614_p1.core.config import settings


@dataclass
class AgentDeps:
    inventory: dict[str, int]
    budget: float
    current_step: int
    opponent_reputation: float
    objective: str = field(default="obtener el mayor valor posible en trueques")
    opponent_alias: str = field(default="desconocido")
    phase: MessagePhase = field(default=MessagePhase.OPENING)
    opponent_inventory: dict[str, int] = field(default_factory=dict)


class OfferResponse(BaseModel):
    msg: str = Field(description="Mensaje directo y específico de negociación.")
    env: dict[str, int] | None = Field(
        default=None,
        description="Items a enviar al oponente. Null si no hay acuerdo cerrado.",
    )

    @field_validator("env", mode="before")
    @classmethod
    def normalize_env(cls, v: Any) -> dict[str, int] | None:
        if not v or v in ("null", "None", ""):
            return None
        if isinstance(v, dict) and not v:
            return None
        return v


# Instrucciones específicas por fase inyectadas en el prompt para que el LLM
# sepa exactamente qué acción se espera sin tener que inferirla del contexto.
_PHASE_GUIDANCE: dict[MessagePhase, str] = {
    MessagePhase.ACCEPTANCE: (
        "ACCION REQUERIDA: El oponente ACABA DE ACEPTAR tu propuesta.\n"
        "Cierra el trato AHORA usando el FORMATO 2: rellena 'env' con los items que prometiste.\n"
        "No propongas nada nuevo. No preguntes. Envía el paquete."
    ),
    MessagePhase.REJECTION: (
        "El oponente ha RECHAZADO tu propuesta.\n"
        "Haz una contraoferta diferente: cambia cantidades o items.\n"
        "No repitas la misma oferta. Sé más flexible para avanzar hacia tu objetivo."
    ),
    MessagePhase.COUNTER_OFFER: (
        "El oponente ha hecho una CONTRAOFERTA.\n"
        "Si te acerca al objetivo, acepta ('acepto, trato hecho').\n"
        "Si no te conviene, contraoferta con algo más favorable para ti."
    ),
    MessagePhase.OPENING: (
        "El oponente está iniciando o continuando conversación.\n"
        "Responde con una propuesta concreta: 'Te doy N X a cambio de M Y. ¿Aceptas?'"
    ),
}


def _reputation_label(score: float) -> str:
    if score < 0.3:
        return "poco fiable — actúa con cautela"
    if score < 0.7:
        return "moderadamente fiable"
    return "muy fiable — puedes ser generoso"


_ollama = AsyncOpenAI(base_url=settings.LLM.BASE_URL, api_key="ollama", timeout=180.0)

_model = OpenAIModel(
    model_name=settings.LLM.MODEL_NAME,
    provider=OpenAIProvider(openai_client=_ollama),
)

agent: Agent[AgentDeps, str] = Agent(model=_model, deps_type=AgentDeps)


@agent.system_prompt
def build_prompt(ctx: RunContext[AgentDeps]) -> str:
    deps = ctx.deps
    inv_lines = (
        "\n".join(f"  - {k}: {v} unidades" for k, v in deps.inventory.items() if v > 0)
        if deps.inventory
        else "  (vacío)"
    )
    opp_lines = (
        "\n".join(
            f"  - {k}: {v} unidades"
            for k, v in deps.opponent_inventory.items()
            if v > 0
        )
        if deps.opponent_inventory
        else "  (desconocido)"
    )
    phase_block = _PHASE_GUIDANCE[deps.phase]
    return (
        f"Eres {settings.AGENT.ALIAS}, un agente de trueque autónomo en una simulación multi-agente.\n\n"
        f"=== OBJETIVO ===\n{deps.objective}\n\n"
        f"=== TU INVENTARIO (solo puedes ofrecer estos items y hasta estas cantidades) ===\n{inv_lines}\n\n"
        f"=== INVENTARIO DEL OPONENTE (solo pide items que el oponente realmente tiene) ===\n{opp_lines}\n\n"
        f"=== NEGOCIACIÓN ===\n"
        f"Oponente: {deps.opponent_alias} ({_reputation_label(deps.opponent_reputation)})\n"
        f"Turno: {deps.current_step}\n\n"
        f"=== SITUACIÓN ACTUAL ===\n{phase_block}\n\n"
        "=== REGLAS GENERALES ===\n"
        "1. Responde ÚNICAMENTE con JSON válido, sin texto adicional.\n"
        "2. Sin saltos de línea reales dentro del JSON.\n"
        "3. FORMATO 1 — proponer/rechazar/contraoferta:\n"
        '   {"msg": "Te doy N X a cambio de M Y. ¿Aceptas?", "env": null}\n'
        "4. FORMATO 2 — cerrar trato (solo cuando el oponente aceptó):\n"
        '   {"msg": "Trato cerrado.", "env": {"item": cantidad}}\n'
        "5. Prioriza siempre los recursos que te acerquen al objetivo.\n"
        "6. Solo ofrece items que realmente tienes en el inventario.\n"
        "7. TODAS las cantidades deben ser números ENTEROS positivos (1, 2, 3...). NUNCA fracciones ni decimales."
    )


def parse_offer(raw: str) -> OfferResponse:
    """Extrae y valida un OfferResponse JSON de la salida en bruto del LLM."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in LLM output: {raw[:120]!r}")
    text = raw[start : end + 1]
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r'"\s*\+\s*"', "", text)
    text = text.replace("\\\n", "").replace("\\n\\", "\\n")
    return OfferResponse.model_validate(json.loads(text, strict=False))
