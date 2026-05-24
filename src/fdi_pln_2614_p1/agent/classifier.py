from enum import Enum


class MessagePhase(str, Enum):
    """Negotiation phase inferred from the opponent's message."""

    OPENING = "opening"
    COUNTER_OFFER = "counter_offer"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"


_ACCEPTANCE = [
    "acepto",
    "sí acepto",
    "trato hecho",
    "de acuerdo",
    "ok trato",
    "vale trato",
    "perfecto",
    "cerrado",
    "agreed",
    "aceptamos",
    "me parece bien",
]
_REJECTION = [
    "no acepto",
    "rechaz",
    "no puedo",
    "no tengo suficiente",
    "no me interesa",
    "imposible",
    "demasiado caro",
    "muy caro",
    "no es suficiente",
    "no trato",
]
_OFFER = [
    "te doy",
    "a cambio de",
    "ofrezco",
    "propongo",
    "cambiaría",
    "trueque",
    "cambio",
]


def classify_message(body: str) -> MessagePhase:
    """Rule-based classifier: determines the negotiation phase from plain text.

    Rejection is checked before acceptance so that "no acepto" does not
    match the "acepto" substring and trigger a false ACCEPTANCE.
    """
    text = body.lower()
    if any(kw in text for kw in _REJECTION):
        return MessagePhase.REJECTION
    if any(kw in text for kw in _ACCEPTANCE):
        return MessagePhase.ACCEPTANCE
    if any(kw in text for kw in _OFFER):
        return MessagePhase.COUNTER_OFFER
    return MessagePhase.OPENING
