DEFAULT_TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 4096

OPENROUTER_REASONING = {
    "none": {"enabled": False},
    "enabled": {"enabled": True}, 
    "minimal": {"effort": "minimal"},
    "low": {"effort": "low"},
    "medium": {"effort": "medium"},
    "high": {"effort": "high"},
    "xhigh": {"effort": "xhigh"},
    "max": {"effort": "max"},
}

OLLAMA_THINK = {
    "default": None,
    "none": False,
    "enabled": True,
    "low": "low",
    "medium": "medium",
    "high": "high",
}

FULLY_VERIFIED_THRESHOLD = 0.95
HARM_POW = 2.426012764375836
MAX_CONTEXT_CHARS = 2000
DEFAULT_HARM_SCORE = 5
