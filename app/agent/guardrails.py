# Disallowed / risky user intent patterns
BLOCKED_KEYWORDS = [
    "ignore previous",
    "ignore instructions",
    "jailbreak",
    "act as a lawyer",
    "act as a government officer",
    "legal advice",
    "medical advice",
    "guarantee approval",
    "guarantee outcome",
    "hack",
    "bypass",
    "exploit",
]


def is_disallowed_question(question: str) -> bool:
    """
    Returns True if the user question violates guardrails.
    """
    q = question.lower()
    return any(keyword in q for keyword in BLOCKED_KEYWORDS)


def guardrail_response() -> str:
    """
    Standard safe response when a question violates guardrails.
    """
    return (
        "I can provide general information about Trust Partners’ services. "
        "For specific advice or case-related guidance, please contact Trust Partners directly."
    )
