import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from loguru import logger

from fdi_pln_2614_p1.agent.classifier import MessagePhase, classify_message
from fdi_pln_2614_p1.agent.core_agent import AgentDeps, agent, parse_offer
from fdi_pln_2614_p1.butler.client import ButlerClient
from fdi_pln_2614_p1.core.config import settings

_historial: dict[str, list] = {}
_reputation: dict[str, float] = {}
_objective: str = "obtener el mayor valor posible en trueques"

_FALLBACK_OBJECTIVE = "obtener el mayor valor posible en trueques"


def _get_reputation(alias: str) -> float:
    return _reputation.get(alias, 0.5)


def _update_reputation(alias: str, phase: MessagePhase) -> None:
    """Nudge reputation up on acceptance, down on rejection."""
    current = _reputation.get(alias, 0.5)
    if phase == MessagePhase.ACCEPTANCE:
        _reputation[alias] = min(1.0, current + 0.15)
    elif phase == MessagePhase.REJECTION:
        _reputation[alias] = max(0.0, current - 0.1)
    else:
        _reputation[alias] = current


async def _fetch_objective(butler: ButlerClient) -> str:
    """Ask the butler for this agent's assigned goal; fall back to a generic one."""
    try:
        info = await butler.get_info()
        obj = info.get("Objetivo") or info.get("Mision") or info.get("Goal")
        if obj is None:
            return _FALLBACK_OBJECTIVE
        if isinstance(obj, dict):
            items = ", ".join(f"{v} {k}" for k, v in obj.items())
            return f"Conseguir mediante trueques: {items}"
        return str(obj)
    except Exception:
        return _FALLBACK_OBJECTIVE


async def _process_message(
    butler: ButlerClient,
    sender: str,
    body: str,
    resources: dict,
    step: int,
) -> None:
    _historial.setdefault(sender, [])

    phase = classify_message(body)
    _update_reputation(sender, phase)
    logger.debug(
        "Phase from {}: {} (reputation now {:.2f})",
        sender,
        phase,
        _get_reputation(sender),
    )

    try:
        opp_info = await butler.get_agent_info(sender)
        opponent_inventory: dict[str, int] = opp_info.get("Recursos", {})
    except Exception:
        opponent_inventory = {}

    deps = AgentDeps(
        inventory=resources,
        budget=float(resources.get("oro", resources.get("gold", 100))),
        current_step=step,
        opponent_reputation=_get_reputation(sender),
        objective=_objective,
        opponent_alias=sender,
        phase=phase,
        opponent_inventory=opponent_inventory,
    )

    result = await agent.run(body, deps=deps, message_history=_historial[sender])
    _historial[sender] = result.all_messages()[-4:]

    try:
        offer = parse_offer(result.output.strip())
        reply_text = offer.msg

        if offer.env:
            logger.info("DEAL with {}! Sending: {}", sender, offer.env)
            await butler.send_package(sender, offer.env)
            _historial[sender] = []

    except Exception as exc:
        logger.warning("Parse error from LLM ({}): {!r:.120}", exc, result.output)
        reply_text = "Lo siento, hay un problema técnico. Hablamos en un momento."
        _historial[sender] = []

    logger.info("-> {} | {}", sender, reply_text)
    await butler.send_message(sender, "Respuesta", reply_text)


async def _mail_loop(butler: ButlerClient) -> None:
    global _objective
    _objective = await _fetch_objective(butler)
    logger.info("Objective loaded: {}", _objective)

    step = 0
    first_contact_done = False

    while True:
        try:
            info = await butler.get_info()
            mailbox: dict = info.get("Buzon") or {}
            resources: dict = info.get("Recursos", {})

            if not mailbox and not first_contact_done:
                agents = await butler.get_agents()
                if agents:
                    target = agents[0]
                    logger.info("Taking initiative with {}", target)
                    _historial.setdefault(target, [])
                    try:
                        t_info = await butler.get_agent_info(target)
                        target_inv: dict[str, int] = t_info.get("Recursos", {})
                    except Exception:
                        target_inv = {}
                    deps = AgentDeps(
                        inventory=resources,
                        budget=float(resources.get("oro", resources.get("gold", 100))),
                        current_step=0,
                        opponent_reputation=_get_reputation(target),
                        objective=_objective,
                        opponent_alias=target,
                        phase=MessagePhase.OPENING,
                        opponent_inventory=target_inv,
                    )
                    result = await agent.run(
                        f"Inicia una negociación con '{target}'. Propón un trato concreto.",
                        deps=deps,
                        message_history=_historial[target],
                    )
                    _historial[target] = result.all_messages()[-4:]
                    try:
                        text = parse_offer(result.output.strip()).msg
                    except Exception:
                        text = f"Hola {target}, tengo {resources}. ¿Qué propones?"
                    logger.info("-> {} (initiative) | {}", target, text)
                    await butler.send_message(target, "Trueque", text)
                    first_contact_done = True

            # Keep only the latest message per sender; silently drop older duplicates
            latest: dict[str, tuple[str, str]] = {}
            for uid, msg in mailbox.items():
                sender: str = msg.get("remi", "")
                if sender in (settings.AGENT.ALIAS, "System", "Sistema"):
                    await butler.delete_message(uid)
                    continue
                if sender in latest:
                    await butler.delete_message(latest[sender][0])
                    logger.debug("Dropped stale duplicate from {}", sender)
                latest[sender] = (uid, msg.get("cuerpo", ""))

            for sender, (uid, body) in latest.items():
                step += 1
                logger.info("<- {} | {}", sender, body[:120])
                await _process_message(butler, sender, body, resources, step)
                await butler.delete_message(uid)

        except Exception as exc:
            logger.error("Loop error: {}", exc)
            await asyncio.sleep(10)
            continue

        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    http = httpx.AsyncClient(timeout=60.0)
    butler = ButlerClient(
        base_url=settings.PLN.BUTLER_ADDRESS,
        alias=settings.AGENT.ALIAS,
        http=http,
    )
    task: asyncio.Task | None = None
    try:
        await butler.register()
        task = asyncio.create_task(_mail_loop(butler))
    except Exception as exc:
        logger.error("Startup failed: {}", exc)

    yield

    if task:
        task.cancel()
    await http.aclose()


app = FastAPI(title="Agente de Trueque", lifespan=lifespan)
