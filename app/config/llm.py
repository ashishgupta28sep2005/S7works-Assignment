"""Shared Groq client construction and credential-safe error classification."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.config.settings import Settings, get_settings


class GroqConfigurationError(RuntimeError):
    """Missing provider setup, with no credential material in the message."""


class GroqRequestError(RuntimeError):
    """Sanitized provider failure category; never includes request/secret text."""

    def __init__(self, category: str, status_code: int | None = None) -> None:
        self.category = category
        # Numeric HTTP status is safe to log; response bodies can contain details
        # that should never be retained in application logs.
        self.status_code = status_code
        super().__init__(safe_groq_error_message(category))


def create_groq_client(settings: Settings | None = None) -> Any:
    """Create one shared Groq client after validating settings without exposing its key."""
    config = settings or get_settings()
    if config.llm_provider != "groq":
        raise GroqConfigurationError(
            f"Unsupported LLM_PROVIDER {config.llm_provider!r}; this phase supports 'groq'."
        )
    if not config.groq_api_key:
        raise GroqConfigurationError("GROQ_API_KEY is missing. Configure it in the ignored .env file.")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise GroqConfigurationError("Install the configured Groq client with: pip install langchain-groq") from exc
    # Retries are disabled: orchestration controls calls and must not duplicate them.
    return ChatGroq(
        model=config.llm_model,
        api_key=config.groq_api_key,
        temperature=0,
        timeout=config.groq_timeout_seconds,
        max_retries=0,
    )


def classify_groq_exception(exc: BaseException) -> str:
    """Map provider and parser exceptions to stable non-sensitive categories."""
    if isinstance(exc, GroqRequestError):
        return exc.category
    if isinstance(exc, ValidationError) or isinstance(exc, (ValueError, TypeError)):
        return "invalid_response"
    name = type(exc).__name__.casefold()
    status = getattr(exc, "status_code", None)
    if "auth" in name or status == 401 or status == 403:
        return "authentication"
    if "ratelimit" in name or "rate_limit" in name or status == 429:
        return "rate_limit"
    if "timeout" in name or "timeout" in str(type(exc)).casefold():
        return "timeout"
    if status == 400:
        return "request_rejected"
    if status == 413:
        return "context_limit"
    if status is not None and 500 <= status <= 599:
        return "provider_error"
    if any(token in name for token in ("outputparser", "jsondecode", "validation", "invalid", "reference")):
        return "invalid_response"
    return "provider_error"


def safe_groq_error_message(category: str) -> str:
    messages = {
        "authentication": "Groq authentication failed; verify GROQ_API_KEY and account access.",
        "rate_limit": "Groq rate limit reached; retry later (no automatic retry was attempted).",
        "timeout": "Groq request timed out; no automatic retry was attempted.",
        "invalid_response": "Groq returned a response that did not satisfy the required structured schema.",
        "request_rejected": "Groq rejected its structured-output request; check the backend log for the HTTP status.",
        "context_limit": "Groq rejected the request because it exceeded the provider context limit.",
        "provider_error": "Groq request failed; provider details were withheld to protect credentials.",
    }
    return messages.get(category, messages["provider_error"])


def invoke_structured(llm: Any, schema: type, messages: list[tuple[str, str]]) -> Any:
    """Run one JSON-schema structured request using Groq's supported schema mode."""
    try:
        if hasattr(llm, "with_structured_output"):
            try:
                structured = llm.with_structured_output(schema, method="json_schema", strict=False)
            except TypeError:
                # Lightweight test doubles may implement only the schema argument.
                structured = llm.with_structured_output(schema)
        else:
            structured = llm
        return structured.invoke(messages)
    except Exception as exc:
        category = classify_groq_exception(exc)
        status_code = getattr(exc, "status_code", None)
        if not isinstance(status_code, int):
            status_code = None
        import logging

        logging.getLogger(__name__).warning(
            "Groq structured request failed (category=%s, http_status=%s).",
            category,
            status_code if status_code is not None else "unavailable",
        )
        raise GroqRequestError(category, status_code=status_code) from None
