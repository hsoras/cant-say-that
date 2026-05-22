# init_llms.py
import os
import sys
import logging
from attackerllm import (
    GeminiAttacker,
    GeminiWithOllamaFallback,
    OllamaAPIError,
    OllamaAttacker,
    OpenRouterAPIError,
    OpenRouterAttacker,
)

logger = logging.getLogger(__name__)


def _ollama_fallback_instance() -> OllamaAttacker:
    """Shared local model used when Gemini returns 503 / other API errors."""
    model = os.environ.get(
        "OLLAMA_FALLBACK_MODEL",
        os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1:8b"),
    )
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return OllamaAttacker(model_name=model, host=host)


def init_llms():
    attacker_backend = os.environ.get("ATTACKER_BACKEND", "ollama").strip().lower()
    judge_backend = os.environ.get("JUDGE_BACKEND", "gemini").strip().lower()

    ollama_fallback: OllamaAttacker | None = None

    def _get_ollama_fallback() -> OllamaAttacker:
        nonlocal ollama_fallback
        if ollama_fallback is None:
            ollama_fallback = _ollama_fallback_instance()
            logger.info(
                "Ollama fallback (for Gemini failures) | model=%s | host=%s",
                ollama_fallback.model_name,
                ollama_fallback.host,
            )
        return ollama_fallback

    try:
        # Build attacker instance
        if attacker_backend == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            attacker_model = os.environ.get("GEMINI_ATTACKER_MODEL", "gemini-3.1-flash-lite-preview")
            if not api_key.strip():
                print(
                    "GEMINI_API_KEY is not set. Export it with your Google AI API key.",
                    file=sys.stderr,
                )
                sys.exit(1)
            logger.info("Attacker backend: Gemini | model=%s", attacker_model)
            attacker_llm = GeminiWithOllamaFallback(
                GeminiAttacker(api_key=api_key, model_name=attacker_model),
                _get_ollama_fallback(),
                role="attacker",
            )
        elif attacker_backend == "ollama":
            attacker_model = os.environ.get("OLLAMA_ATTACKER_MODEL", "dolphin-phi")
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            logger.info("Attacker backend: Ollama | model=%s | host=%s", attacker_model, host)
            attacker_llm = OllamaAttacker(model_name=attacker_model, host=host)
        elif attacker_backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            attacker_model = os.environ.get("OPENROUTER_ATTACKER_MODEL", "anthropic/claude-sonnet-4")
            if not api_key.strip():
                print(
                    "OPENROUTER_API_KEY is not set. Export it with your OpenRouter API key.",
                    file=sys.stderr,
                )
                sys.exit(1)
            logger.info("Attacker backend: OpenRouter | model=%s", attacker_model)
            attacker_llm = OpenRouterAttacker(api_key=api_key, model_name=attacker_model)
        else:
            print(
                f"Unknown ATTACKER_BACKEND={attacker_backend!r}. Use 'gemini', 'ollama', or 'openrouter'.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Build judge instance
        if judge_backend == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            judge_model = os.environ.get("GEMINI_JUDGE_MODEL", "gemini-3.1-flash-lite-preview")
            if not api_key.strip():
                print(
                    "GEMINI_API_KEY is not set. Export it with your Google AI API key.",
                    file=sys.stderr,
                )
                sys.exit(1)
            logger.info("Judge backend: Gemini | model=%s", judge_model)
            judge_llm = GeminiWithOllamaFallback(
                GeminiAttacker(api_key=api_key, model_name=judge_model),
                _get_ollama_fallback(),
                role="judge",
            )
        elif judge_backend == "ollama":
            judge_model = os.environ.get("OLLAMA_JUDGE_MODEL", "llama3.1:8b")
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            logger.info("Judge backend: Ollama | model=%s | host=%s", judge_model, host)
            judge_llm = OllamaAttacker(model_name=judge_model, host=host)
        elif judge_backend == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            judge_model = os.environ.get("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini")
            if not api_key.strip():
                print(
                    "OPENROUTER_API_KEY is not set. Export it with your OpenRouter API key.",
                    file=sys.stderr,
                )
                sys.exit(1)
            logger.info("Judge backend: OpenRouter | model=%s", judge_model)
            judge_llm = OpenRouterAttacker(api_key=api_key, model_name=judge_model)
        else:
            print(
                f"Unknown JUDGE_BACKEND={judge_backend!r}. Use 'gemini', 'ollama', or 'openrouter'.",
                file=sys.stderr,
            )
            sys.exit(1)

        return attacker_llm, judge_llm
    except (OllamaAPIError, OpenRouterAPIError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)