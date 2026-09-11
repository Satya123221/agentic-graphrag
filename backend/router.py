"""Deterministic question router for the local chatbot.

Sits in front of the existing Agentic GraphRAG planner. Does not rewrite
planning, RAG retrieval, or GraphRAG ranking. Does not call Ollama to classify.
"""

import ast
import operator
import re

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

ROUTE_AGENTIC = "agentic_graphrag"
ROUTE_GENERAL = "general_llm"
ROUTE_CALCULATION = "calculation"
ROUTE_CURRENT = "current_web"

ROUTE_LABELS = {
    ROUTE_AGENTIC: "Agentic GraphRAG",
    ROUTE_GENERAL: "General LLM",
    ROUTE_CALCULATION: "Calculation",
    ROUTE_CURRENT: "Current/Web",
}

CURRENT_WEB_ANSWER = (
    "This question asks for current or time-sensitive information. "
    "Web search is not available in this deployment, so I cannot provide "
    "live or up-to-date answers."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
MONTH_ALT = (
    r"January|February|March|April|May|June|July|"
    r"August|September|October|November|December"
)
DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}\s+{MONTH_ALT}|\b{MONTH_ALT}\s+\d{{1,2}})\b",
    re.IGNORECASE,
)

# Longest phrases first so "table tennis" wins over "tennis".
SPORT_PHRASES = tuple(sorted((
    "alpine skiing", "artistic gymnastics", "artistic swimming",
    "athletics", "badminton", "baseball", "basketball", "beach volleyball",
    "biathlon", "bobsleigh", "bobsled", "boxing", "canoe sprint", "canoeing",
    "cross-country skiing", "curling", "cycling", "diving", "equestrian",
    "fencing", "figure skating", "football", "freestyle skiing", "golf",
    "gymnastics", "handball", "hockey", "ice hockey", "judo", "karate",
    "kayak", "luge", "marathon", "modern pentathlon", "nordic combined",
    "rhythmic gymnastics", "rowing", "rugby", "sailing", "shooting",
    "short track", "skateboarding", "skeleton", "ski jumping", "snowboarding",
    "softball", "speed skating", "sport climbing", "surfing",
    "synchronized swimming", "table tennis", "taekwondo", "tennis",
    "trampoline", "triathlon", "volleyball", "water polo", "weightlifting",
    "wrestling", "race walking", "kilometres walk", "kilometers walk",
), key=len, reverse=True))
SPORT_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in SPORT_PHRASES) + r")\b",
    re.IGNORECASE,
)

OLYMPIC_RE = re.compile(
    r"\b(olympics?|olympic games|paralympics?)\b",
    re.IGNORECASE,
)
SUMMER_WINTER_GAMES_RE = re.compile(
    r"\b(?:\d{4}\s+)?(?:summer|winter)\s+games\b",
    re.IGNORECASE,
)
HELD_AT_RE = re.compile(r"\bheld at\b", re.IGNORECASE)
CORPUS_RE = re.compile(
    r"\baccording to\b.{0,48}\bcorpus\b|\bprovided corpus\b",
    re.IGNORECASE,
)
EVENT_ASK_RE = re.compile(
    r"\b("
    r"gold medal|won gold|silver medal|bronze medal|"
    r"which athletes?|which event|who won|who participated|who represented|"
    r"how many.{0,48}events?|number of (?:competitors|nations|participants)|"
    r"highest number of|fewest number of|lowest number of|"
    r"weight class|\d+\s*kg"
    r")\b",
    re.IGNORECASE,
)
CURRENT_RE = re.compile(
    r"\b("
    r"today|tonight|this morning|this afternoon|this evening|"
    r"this week|this month|right now|as of now|as of today|"
    r"currently|latest news|breaking news|live scores?|live results|"
    r"weather (?:today|now|forecast)|stock prices?|"
    r"who is the current|"
    r"current (?:president|prime minister|champion|leader|ceo|score|weather|news)"
    r")\b",
    re.IGNORECASE,
)

MATH_WRAP_RE = re.compile(
    r"^(?:what(?:'s| is)|calculate|compute|evaluate|how much is)\s+",
    re.IGNORECASE,
)
WORD_OPS = (
    (re.compile(r"\bmultiplied by\b", re.IGNORECASE), "*"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    (re.compile(r"\bdivided by\b", re.IGNORECASE), "/"),
)
X_MULT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)")
EXPR_CHARS_RE = re.compile(r"^[\d\.\s\+\-\*/\(\)%]+$")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

GENERAL_SYSTEM_PROMPT = """You are a helpful general-knowledge assistant.
Answer using your own knowledge.
This is not an Olympic corpus question. Do not retrieve, cite, or rely on Olympic documents.
Be concise and factual.
/no_think"""


def _norm_math_question(question):
    text = (question or "").strip().rstrip(" ?!.")
    text = MATH_WRAP_RE.sub("", text).strip()
    for pattern, repl in WORD_OPS:
        text = pattern.sub(repl, text)
    text = X_MULT_RE.sub(r"\1 * \2", text)
    text = text.replace("^", "**")
    return " ".join(text.split())


def parse_math_expression(question):
    """Return a safe arithmetic expression, or None if this is not a calc question."""
    text = _norm_math_question(question)
    if not text:
        return None
    display = text.replace("**", "*")
    if not EXPR_CHARS_RE.fullmatch(display):
        return None
    if not re.search(r"\d", text):
        return None
    compact = re.sub(r"\s+", "", text).replace("**", "*")
    if not re.search(r"\d[\+\-\*/%]+\d", compact):
        return None
    return text


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def evaluate_calculation(expression):
    tree = ast.parse(expression, mode="eval")
    value = _eval_ast(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _is_olympic_question(question):
    raw = question or ""
    if OLYMPIC_RE.search(raw) or SUMMER_WINTER_GAMES_RE.search(raw):
        return True
    if CORPUS_RE.search(raw):
        return True
    has_date = bool(DATE_RE.search(raw) or YEAR_RE.search(raw))
    if HELD_AT_RE.search(raw) and has_date:
        return True
    sport = bool(SPORT_RE.search(raw))
    event_ask = bool(EVENT_ASK_RE.search(raw))
    if sport and (event_ask or has_date):
        return True
    if event_ask and has_date:
        return True
    return False


def _is_current_question(question):
    return bool(CURRENT_RE.search(question or ""))


def classify(question):
    """Return a deterministic routing decision. Never calls Ollama."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question is required")
    expression = parse_math_expression(text)
    if expression is not None:
        return {
            "route": ROUTE_CALCULATION,
            "route_label": ROUTE_LABELS[ROUTE_CALCULATION],
            "reason": "Deterministic arithmetic expression.",
            "expression": expression,
        }
    olympic = _is_olympic_question(text)
    current = _is_current_question(text)
    if current and not (olympic and YEAR_RE.search(text)):
        return {
            "route": ROUTE_CURRENT,
            "route_label": ROUTE_LABELS[ROUTE_CURRENT],
            "reason": "Current or time-sensitive question; web search is not available.",
            "expression": None,
        }
    if olympic:
        return {
            "route": ROUTE_AGENTIC,
            "route_label": ROUTE_LABELS[ROUTE_AGENTIC],
            "reason": "Olympic or corpus question; use Agentic GraphRAG.",
            "expression": None,
        }
    return {
        "route": ROUTE_GENERAL,
        "route_label": ROUTE_LABELS[ROUTE_GENERAL],
        "reason": "General knowledge question outside the Olympic corpus.",
        "expression": None,
    }


def _strip_think(text):
    return THINK_RE.sub("", text or "").strip()


def ask_general_llm(question):
    """Answer with local Ollama. Does not retrieve Olympic corpus documents."""
    prompt = (
        f"{GENERAL_SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_predict": 512,
        },
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=600,
    )
    if response.status_code == 400:
        payload.pop("think", None)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=600,
        )
    response.raise_for_status()
    body = response.json()
    return _strip_think(body.get("response", ""))
